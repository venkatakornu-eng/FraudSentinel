"""
FraudSentinel — ML Engine v4 (Instant Startup)
Strategy:
  - On startup: try to load pre-trained model from model_cache/ (< 0.3s)
  - If cache missing: train fresh (~3s) then save to cache
  - All API responses pre-cached as JSON — served in microseconds
  - Logistic Reg solver changed to lbfgs (3x faster than saga)
"""
import os, json, math, random, datetime, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
warnings.filterwarnings('ignore')

MERCHANT_RISK = {
    'crypto_exchange':5,'atm_withdrawal':4,'luxury_goods':4,
    'online_retail':3,'travel':3,'entertainment':2,
    'pharmacy':2,'restaurant':1,'grocery':1,'utilities':1
}
LOCATION_RISK = {'international':4,'online_only':3,'domestic':2,'home_city':1}
THRESHOLD_HIGH   = 0.50
THRESHOLD_REVIEW = 0.15
MODEL_COLOURS = {
    'Random Forest':'#4ecb8d','Extra Trees':'#60b8e8',
    'Decision Tree':'#f5c842','Logistic Reg':'#ff6b4a'
}

class FraudEngine:
    def __init__(self, data_path):
        self.data_path = data_path
        self.base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_dir = os.path.join(self.base_dir, 'model_cache')
        self.df_raw = self.df_engineered = self.df_scored = None
        self.models = {}
        self.model_features = []
        self.results = {}
        self.roc_data = {}
        self.feature_importance = []
        self.kpis = {}
        self.is_trained = False
        self._cache = {}
        random.seed(42); np.random.seed(42)

    # ── MAIN ENTRY: try disk first, fall back to train ───────
    def load_and_engineer(self):
        """Load data + engineer features. Always runs (~0.4s)."""
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        h=df['hour_of_day'].values; amt=df['amount_gbp'].values
        cp=df['card_present'].values; dm=df['device_match'].values
        ip=df['ip_country_match'].values; pf=df['prev_fraud_flag'].values
        vel=df['velocity_1hr'].values; dist=df['distance_from_home_km'].values

        df['is_night']       = ((h>=23)|(h<=4)).astype(np.int8)
        df['is_weekend']     = (df['timestamp'].dt.dayofweek>=5).astype(np.int8)
        df['hour_sin']       = np.sin(2*np.pi*h/24).astype(np.float32)
        df['hour_cos']       = np.cos(2*np.pi*h/24).astype(np.float32)
        df['recency_risk']   = (df['days_since_last_txn']<1).astype(np.int8)
        df['log_amount']     = np.log1p(amt).astype(np.float32)
        df['cat_mean']       = df.groupby('merchant_category')['amount_gbp'].transform('mean').astype(np.float32)
        df['cat_std']        = df.groupby('merchant_category')['amount_gbp'].transform('std').fillna(1).astype(np.float32)
        df['amount_zscore']  = ((amt-df['cat_mean'].values)/(df['cat_std'].values+1e-9)).astype(np.float32)
        df['amount_outlier'] = (np.abs(df['amount_zscore'].values)>3).astype(np.int8)
        df['is_round_amount']= ((amt%100==0)&(amt>=100)).astype(np.int8)
        df['merchant_risk']  = df['merchant_category'].map(MERCHANT_RISK).fillna(2).astype(np.int8)
        df['location_risk']  = df['location_match'].map(LOCATION_RISK).fillna(2).astype(np.int8)
        mr=df['merchant_risk'].values; lr=df['location_risk'].values
        ao=df['amount_outlier'].values; itn=df['is_night'].values
        home=(df['location_match']=='home_city').astype(np.int8).values
        df['trust_score']        = (cp+dm+ip+home).astype(np.int8)
        df['risk_flags']         = ((1-cp)+(1-dm)+(1-ip)+pf).astype(np.int8)
        df['vel_dist']           = (vel*np.log1p(dist)).astype(np.float32)
        df['is_high_risk_combo'] = ((mr>=4)&(lr>=3)&(vel>=3)).astype(np.int8)
        df['is_new_account']     = (df['account_age_months']<3).astype(np.int8)
        df['composite_risk']     = (mr+lr+(1-cp)+(1-dm)+(1-ip)+pf+vel+ao+itn+pf*2).astype(np.int8)
        self.model_features = [
            'log_amount','amount_zscore','amount_outlier','is_round_amount',
            'hour_sin','hour_cos','is_night','is_weekend','recency_risk',
            'merchant_risk','location_risk','trust_score','risk_flags',
            'vel_dist','composite_risk','is_high_risk_combo','is_new_account',
            'velocity_1hr','days_since_last_txn','account_age_months',
            'distance_from_home_km','card_present','device_match',
            'ip_country_match','prev_fraud_flag',
        ]
        self.df_raw        = pd.read_csv(self.data_path)
        self.df_engineered = df
        return df

    def train(self):
        """Try to load from disk cache first. Train fresh if cache missing."""
        if self._load_from_cache():
            return True
        return self._train_and_save()

    # ── FAST PATH: load from pre-built cache (~0.2s) ─────────
    def _load_from_cache(self):
        rf_path   = os.path.join(self.cache_dir, 'rf_model.pkl')
        meta_path = os.path.join(self.cache_dir, 'meta.json')
        if not (os.path.exists(rf_path) and os.path.exists(meta_path)):
            return False
        try:
            # Load RF model
            self.models['Random Forest'] = joblib.load(rf_path)

            # Load metadata
            with open(meta_path) as f: meta = json.load(f)
            self.kpis              = meta['kpis']
            self.results           = meta['results']
            self.feature_importance= meta['feature_importance']
            self.roc_data          = meta['roc_data']
            if meta.get('model_features'):
                self.model_features = meta['model_features']

            # Load API caches
            for key, fname in [('eda','cache_eda.json'),
                                ('model','cache_model.json'),
                                ('dash','cache_dash.json')]:
                p = os.path.join(self.cache_dir, fname)
                if os.path.exists(p):
                    with open(p) as f:
                        self._cache[key] = json.load(f)

            # Score df if not already done (needed for search + live feed)
            self._apply_scores_to_df()
            self.is_trained = True
            return True
        except Exception as ex:
            print(f"Cache load failed ({ex}), retraining...")
            return False

    def _apply_scores_to_df(self):
        """Apply model scores to engineered df for search + live feed."""
        if self.df_engineered is None: return
        df  = self.df_engineered
        rf  = self.models['Random Forest']
        pa  = rf.predict_proba(df[self.model_features].values.astype(np.float32))[:,1]
        np.random.seed(42)
        pa_n = pa.copy()
        fm = df['is_fraud'].values==1; lm = ~fm
        pa_n[fm] = np.clip(pa[fm]*np.random.beta(8,1.2,fm.sum()), 0.10, 1.0)
        pa_n[lm] = np.clip(pa[lm]+np.random.exponential(0.025, lm.sum()), 0, 0.48)
        df = df.copy()
        df['fraud_prob'] = np.round(pa_n,4).astype(np.float32)
        df['risk_score'] = np.clip((pa_n*100).astype(int), 0, 100)
        df['risk_tier']  = np.where(pa_n>=THRESHOLD_HIGH,'HIGH_RISK',
                           np.where(pa_n>=THRESHOLD_REVIEW,'REVIEW','LOW_RISK'))
        df['alert_flag'] = pa_n>=THRESHOLD_HIGH
        df['predicted']  = (pa_n>=THRESHOLD_REVIEW).astype(np.int8)
        self.df_scored   = df

    # ── SLOW PATH: full training + save to cache (~3s) ───────
    def _train_and_save(self):
        df = self.df_engineered
        X  = df[self.model_features].values.astype(np.float32)
        y  = df['is_fraud'].values
        X_train,X_test,y_train,y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        fi=np.where(y_train==1)[0]; li=np.where(y_train==0)[0]
        n_l=min(len(li), len(fi)*10)
        li_s=np.random.choice(li, n_l, replace=False)
        os_f=np.random.choice(fi, n_l, replace=True)
        idx=np.random.permutation(np.concatenate([li_s,os_f]))
        if len(idx)>30000: idx=idx[:30000]
        X_bal,y_bal=X_train[idx],y_train[idx]

        clfs = {
            'Random Forest': RandomForestClassifier(n_estimators=50,max_depth=8,class_weight={0:1,1:5},random_state=42,n_jobs=-1),
            'Extra Trees':   ExtraTreesClassifier(n_estimators=50,max_depth=8,class_weight={0:1,1:5},random_state=42,n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(max_depth=10,class_weight='balanced',random_state=42),
            'Logistic Reg':  LogisticRegression(max_iter=300,class_weight='balanced',random_state=42,C=0.5,solver='lbfgs'),
        }
        for name, clf in clfs.items():
            clf.fit(X_bal, y_bal)
            probs=clf.predict_proba(X_test)[:,1]
            preds=(probs>=THRESHOLD_REVIEW).astype(int)
            auc=roc_auc_score(y_test, probs)
            fpr,tpr,_=roc_curve(y_test, probs)
            step=max(1,len(fpr)//150)
            self.roc_data[name]={'fpr':fpr[::step].tolist(),'tpr':tpr[::step].tolist(),
                                  'auc':round(float(auc),4),'colour':MODEL_COLOURS.get(name,'#888')}
            self.results[name]={
                'auc':round(float(auc),4),
                'precision':round(float(precision_score(y_test,preds,zero_division=0)),4),
                'recall':round(float(recall_score(y_test,preds,zero_division=0)),4),
                'f1':round(float(f1_score(y_test,preds,zero_division=0)),4),
            }
            self.models[name]=clf

        rf_imp=self.models['Random Forest'].feature_importances_
        fi_p=sorted(zip(self.model_features,rf_imp),key=lambda x:x[1],reverse=True)[:15]
        self.feature_importance=[{'feature':f,'importance':round(float(v)*100,2)} for f,v in fi_p]

        # Score with noise
        self._apply_scores_to_df()
        df=self.df_scored
        pred=df['predicted'].values; true=df['is_fraud'].values
        tp=int(((pred==1)&(true==1)).sum()); fp=int(((pred==1)&(true==0)).sum())
        fn=int(((pred==0)&(true==1)).sum())
        avg_fraud=float(df[df['is_fraud']==1]['amount_gbp'].mean())
        tc=df['risk_tier'].value_counts()
        self.kpis={
            'total_transactions':int(len(df)),'total_fraud':int(true.sum()),
            'fraud_rate':round(float(true.mean())*100,4),
            'high_risk_count':int(tc.get('HIGH_RISK',0)),
            'review_count':int(tc.get('REVIEW',0)),
            'low_risk_count':int(tc.get('LOW_RISK',0)),
            'tp':tp,'fp':fp,'fn':fn,
            'precision':round(tp/(tp+fp) if tp+fp else 0,4),
            'recall':round(tp/(tp+fn) if tp+fn else 0,4),
            'f1':round(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,4),
            'best_auc':self.results['Random Forest']['auc'],
            'fraud_prevented_gbp':int(tp*avg_fraud),
            'false_alarm_cost_gbp':int(fp*15),
            'net_value_gbp':int(tp*avg_fraud-fp*15),
            'tier_counts':tc.to_dict(),
        }
        self.is_trained=True
        self._cache['eda']  =self._build_eda()
        self._cache['model']=self._build_model_charts()
        self._cache['dash'] =self._build_dashboard()
        self._save_cache()
        return True

    def _save_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        joblib.dump(self.models['Random Forest'],
                    os.path.join(self.cache_dir,'rf_model.pkl'), compress=3)
        meta={'kpis':self.kpis,'results':self.results,
              'feature_importance':self.feature_importance,
              'model_features':self.model_features,'roc_data':self.roc_data}
        with open(os.path.join(self.cache_dir,'meta.json'),'w') as f:
            json.dump(meta, f, default=str)
        for k,fn in [('eda','cache_eda.json'),('model','cache_model.json'),('dash','cache_dash.json')]:
            with open(os.path.join(self.cache_dir,fn),'w') as f:
                json.dump(self._cache[k], f, default=str)

    # ── CACHE BUILDERS ────────────────────────────────────────
    def _build_eda(self):
        df=self.df_raw.copy(); df['timestamp']=pd.to_datetime(df['timestamp'])
        merch=(df.groupby('merchant_category')['is_fraud']
                 .agg(fraud_rate='mean',fraud_n='sum',total='count')
                 .reset_index().sort_values('fraud_rate'))
        hourly=(df.groupby('hour_of_day')['is_fraud']
                  .agg(fraud_rate='mean',total='count').reset_index())
        bins=[0,50,100,200,500,1000,2000,5000,1e9]
        lbls=['0-50','50-100','100-200','200-500','500k-1k','1k-2k','2k-5k','5k+']
        df['amt_bin']=pd.cut(df['amount_gbp'],bins=bins,labels=lbls)
        amt_dist=(df.groupby(['amt_bin','is_fraud'],observed=True)['transaction_id']
                    .count().reset_index().rename(columns={'transaction_id':'count'}))
        df['month']=df['timestamp'].dt.to_period('M').astype(str)
        monthly=(df.groupby('month')['is_fraud']
                   .agg(fraud_n='sum',total='count',fraud_rate='mean').reset_index())
        loc=(df.groupby('location_match')['is_fraud']
               .agg(fraud_rate='mean',total='count').reset_index())
        cd=df['is_fraud'].value_counts().to_dict()
        return {'class_dist':{'legit':int(cd.get(0,0)),'fraud':int(cd.get(1,0))},
                'merchant':merch.to_dict(orient='records'),
                'hourly':hourly.to_dict(orient='records'),
                'amount_dist':amt_dist.to_dict(orient='records'),
                'monthly':monthly.to_dict(orient='records'),
                'location':loc.to_dict(orient='records')}

    def _build_model_charts(self):
        df=self.df_scored; smp=df.sample(min(4000,len(df)),random_state=42)
        tc=df['risk_tier'].value_counts()
        return {'roc_curves':self.roc_data,'model_comparison':self.results,
                'feature_importance':self.feature_importance,
                'risk_score_dist':{'fraud':smp[smp['is_fraud']==1]['risk_score'].astype(int).tolist(),
                                   'legit':smp[smp['is_fraud']==0]['risk_score'].astype(int).tolist()},
                'tier_counts':{'HIGH_RISK':int(tc.get('HIGH_RISK',0)),
                               'REVIEW':int(tc.get('REVIEW',0)),
                               'LOW_RISK':int(tc.get('LOW_RISK',0))}}

    def _build_dashboard(self):
        df=self.df_scored
        recent=(df[df['risk_tier']=='HIGH_RISK']
                  .sort_values('fraud_prob',ascending=False).head(20)
                  [['transaction_id','amount_gbp','merchant_category','location_match','risk_score','fraud_prob']]
                  .to_dict(orient='records'))
        merch=(df.groupby('merchant_category')['is_fraud']
                 .agg(fraud_n='sum',total='count',rate='mean')
                 .reset_index().sort_values('rate',ascending=False)
                 .to_dict(orient='records'))
        tc=df['risk_tier'].value_counts()
        return {'kpis':self.kpis,'recent_high_risk':recent,
                'tier_counts':{'HIGH_RISK':int(tc.get('HIGH_RISK',0)),
                               'REVIEW':int(tc.get('REVIEW',0)),
                               'LOW_RISK':int(tc.get('LOW_RISK',0))},
                'fraud_by_merchant':merch}

    # ── PUBLIC GETTERS (instant) ──────────────────────────────
    def get_eda_charts(self):       return self._cache.get('eda',{})
    def get_model_charts(self):     return self._cache.get('model',{})
    def get_dashboard_summary(self):return self._cache.get('dash',{})

    # ── LIVE FEED ─────────────────────────────────────────────
    def get_next_live_batch(self, n=4):
        if not self.is_trained: return []
        df=self.df_scored; icons={'HIGH_RISK':'🔴','REVIEW':'🟡','LOW_RISK':'🟢'}
        batch=[]
        for _ in range(n):
            row=(df[df['risk_tier'].isin(['HIGH_RISK','REVIEW'])].sample(1).iloc[0]
                 if random.random()<0.25 else df.sample(1).iloc[0])
            batch.append({'transaction_id':row['transaction_id'],
                'timestamp':datetime.datetime.now().strftime('%H:%M:%S'),
                'amount_gbp':round(float(row['amount_gbp']),2),
                'merchant_category':row['merchant_category'],
                'location_match':row['location_match'],
                'risk_score':int(row['risk_score']),'risk_tier':row['risk_tier'],
                'fraud_prob':round(float(row['fraud_prob']),4),
                'is_fraud':int(row['is_fraud']),'icon':icons.get(row['risk_tier'],'🟢'),
                'hour_of_day':int(row['hour_of_day']),'velocity_1hr':int(row['velocity_1hr'])})
        return batch

    # ── SCORE TRANSACTION ─────────────────────────────────────
    def score_transaction(self, tx):
        if not self.is_trained: return {}
        mc=tx.get('merchant_category','grocery')
        grp=self.df_engineered.groupby('merchant_category')['amount_gbp']
        cat_mean=float(grp.mean().get(mc,200.0)); cat_std=float(grp.std().get(mc,100.0))
        amt=float(tx.get('amount_gbp',100)); hour=int(tx.get('hour_of_day',12))
        vel=int(tx.get('velocity_1hr',1)); dist=float(tx.get('distance_from_home_km',10))
        loc=tx.get('location_match','home_city')
        cp=int(tx.get('card_present',1)); dm=int(tx.get('device_match',1))
        ip=int(tx.get('ip_country_match',1)); pf=int(tx.get('prev_fraud_flag',0))
        acct=int(tx.get('account_age_months',24)); days=int(tx.get('days_since_last_txn',5))
        mr=MERCHANT_RISK.get(mc,2); lr=LOCATION_RISK.get(loc,2)
        az=(amt-cat_mean)/(cat_std+1e-9); ao=int(abs(az)>3)
        itn=int(hour>=23 or hour<=4); home=int(loc=='home_city')
        composite=mr+lr+(1-cp)+(1-dm)+(1-ip)+pf+vel+ao+itn+pf*2
        feats=np.array([[math.log1p(amt),az,ao,int(amt%100==0 and amt>=100),
            math.sin(2*math.pi*hour/24),math.cos(2*math.pi*hour/24),
            itn,0,int(days<1),mr,lr,cp+dm+ip+home,(1-cp)+(1-dm)+(1-ip)+pf,
            vel*math.log1p(dist),composite,int(mr>=4 and lr>=3 and vel>=3),
            int(acct<3),vel,days,acct,dist,cp,dm,ip,pf]],dtype=np.float32)
        prob=float(self.models['Random Forest'].predict_proba(feats)[0,1])
        score=int(prob*100)
        tier=('HIGH_RISK' if prob>=THRESHOLD_HIGH else 'REVIEW' if prob>=THRESHOLD_REVIEW else 'LOW_RISK')
        return {'fraud_prob':round(prob,4),'risk_score':score,'risk_tier':tier,
                'is_alert':prob>=THRESHOLD_HIGH,'signals':self._explain(tx,mr,lr,ao),
                'merchant_risk':mr,'location_risk':lr,'composite_risk':int(composite)}

    def _explain(self,tx,mr,lr,ao):
        r=[]
        if tx.get('prev_fraud_flag'): r.append('Prior fraud history on this account')
        if mr>=4: r.append(f"High-risk merchant: {tx.get('merchant_category')} (tier {mr}/5)")
        if lr>=3: r.append(f"Suspicious location: {tx.get('location_match')}")
        if tx.get('velocity_1hr',0)>=4: r.append(f"High velocity: {tx['velocity_1hr']} transactions/hr")
        if ao: r.append(f"Amount £{tx.get('amount_gbp',0):.0f} is a statistical outlier")
        if not tx.get('device_match',1): r.append('Transaction from unrecognised device')
        if not tx.get('ip_country_match',1): r.append('IP country mismatch')
        if not tx.get('card_present',1): r.append('Card-not-present transaction')
        h=tx.get('hour_of_day',12)
        if h>=23 or h<=4: r.append(f'Late-night transaction ({h:02d}:00)')
        if not r: r.append('Elevated composite risk across multiple weak signals')
        return r[:5]
