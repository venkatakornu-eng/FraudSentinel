"""FraudSentinel — Flask API Server v3"""
import sys,os,json,gzip,datetime,threading
sys.path.insert(0,os.path.dirname(__file__))
from flask import Flask,jsonify,request,render_template,Response
from backend.engine import FraudEngine

app=Flask(__name__,template_folder='templates',static_folder='static')
app.config['JSON_SORT_KEYS']=False

DATA_PATH=os.path.join(os.path.dirname(__file__),'data','fraud_transactions.csv')
engine=FraudEngine(DATA_PATH)
feedback_store=[]
status={'done':False,'progress':0,'message':'Starting…'}

@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin']='*'
    r.headers['Access-Control-Allow-Headers']='Content-Type'
    r.headers['Access-Control-Allow-Methods']='GET,POST,OPTIONS'
    return r

def gz(data):
    body=json.dumps(data,default=str).encode()
    if 'gzip' in request.headers.get('Accept-Encoding',''):
        return Response(gzip.compress(body,compresslevel=1),
                        mimetype='application/json',
                        headers={'Content-Encoding':'gzip'})
    return Response(body,mimetype='application/json')

def _train():
    global status
    try:
        status={'done':False,'progress':5,'message':'Loading data…'}
        engine.load_and_engineer()
        status={'done':False,'progress':20,'message':'Engineering features…'}
        engine.train()
        status={'done':True,'progress':100,'message':'Ready'}
    except Exception as ex:
        status={'done':False,'progress':-1,'message':str(ex)}

threading.Thread(target=_train,daemon=True).start()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/status')
def api_status(): return jsonify(status)

@app.route('/api/dashboard')
def api_dashboard():
    if not engine.is_trained: return jsonify({'error':'training'}),202
    return gz(engine.get_dashboard_summary())

@app.route('/api/eda')
def api_eda():
    if not engine.is_trained: return jsonify({'error':'training'}),202
    return gz(engine.get_eda_charts())

@app.route('/api/models')
def api_models():
    if not engine.is_trained: return jsonify({'error':'training'}),202
    return gz(engine.get_model_charts())

@app.route('/api/live-feed')
def api_live():
    if not engine.is_trained: return jsonify([])
    return jsonify(engine.get_next_live_batch(int(request.args.get('n',4))))

@app.route('/api/score',methods=['POST','OPTIONS'])
def api_score():
    if request.method=='OPTIONS': return '',204
    if not engine.is_trained: return jsonify({'error':'not ready'}),202
    return jsonify(engine.score_transaction(request.get_json(force=True)))

@app.route('/api/search')
def api_search():
    if not engine.is_trained: return jsonify({'total':0,'page':1,'data':[]})
    q=request.args.get('q','').lower(); tier=request.args.get('tier','')
    page=max(1,int(request.args.get('page',1))); per=min(50,int(request.args.get('per',25)))
    df=engine.df_scored
    if q:
        df=df[df['transaction_id'].str.lower().str.contains(q,na=False)|
              df['merchant_category'].str.lower().str.contains(q,na=False)]
    if tier: df=df[df['risk_tier']==tier]
    df=df.sort_values('risk_score',ascending=False)
    total=len(df); start=(page-1)*per
    cols=['transaction_id','timestamp','amount_gbp','merchant_category',
          'location_match','risk_score','risk_tier','fraud_prob',
          'is_fraud','predicted','hour_of_day','velocity_1hr',
          'distance_from_home_km','composite_risk']
    return gz({'total':total,'page':page,'data':df.iloc[start:start+per][cols].to_dict(orient='records')})

@app.route('/api/llm-narrative',methods=['POST','OPTIONS'])
def api_narrative():
    if request.method=='OPTIONS': return '',204
    d=request.get_json(force=True)
    tx=d.get('transaction',{}); signals=d.get('signals',[]); api_key=d.get('api_key','').strip()
    if len(api_key)>10:
        try:
            import urllib.request as ur,json as _j
            payload={"model":"claude-haiku-4-5-20251001","max_tokens":220,
                "system":"You are FraudSentinel, a UK fintech fraud analyst AI. Be concise, professional, actionable.",
                "messages":[{"role":"user","content":
                    f"Transaction: {tx.get('transaction_id','?')} | £{tx.get('amount_gbp',0):.2f} | "
                    f"{tx.get('merchant_category','?')} | {tx.get('location_match','?')} | "
                    f"Score: {tx.get('risk_score',0)}/100 | {tx.get('risk_tier','?')}\n"
                    f"Signals:\n"+"\n".join(f"{i+1}. {s}" for i,s in enumerate(signals))+
                    "\n\nWrite a professional 2-3 sentence fraud alert. State risk, cite top factors, recommend Block/Review/Monitor."}]}
            req=ur.Request('https://api.anthropic.com/v1/messages',
                data=_j.dumps(payload).encode(),
                headers={'x-api-key':api_key,'anthropic-version':'2023-06-01','content-type':'application/json'})
            with ur.urlopen(req,timeout=10) as resp:
                r=_j.loads(resp.read())
                return jsonify({'narrative':r['content'][0]['text'],'source':'Claude LLM'})
        except: pass
    tier=tx.get('risk_tier','LOW_RISK')
    action={'HIGH_RISK':'BLOCK immediately and verify cardholder.',
            'REVIEW':'Queue for analyst review within 30 minutes.',
            'LOW_RISK':'Monitor passively.'}.get(tier,'Monitor.')
    sig='; '.join(signals[:3]) if signals else 'elevated composite risk'
    return jsonify({'narrative':
        f"FraudSentinel flagged {tx.get('transaction_id','?')} (£{tx.get('amount_gbp',0):.2f} at "
        f"{tx.get('merchant_category','?')}) with score {tx.get('risk_score',0)}/100 ({tier}). "
        f"Primary signals: {sig}. {action}",
        'source':'Rule-based'})

@app.route('/api/feedback',methods=['POST','OPTIONS'])
def api_fb_post():
    if request.method=='OPTIONS': return '',204
    d=request.get_json(force=True)
    entry={'id':len(feedback_store)+1,'timestamp':datetime.datetime.now().isoformat(),
           **{k:d.get(k,'') for k in ['name','role','rating','category','message']}}
    feedback_store.append(entry)
    return jsonify({'success':True,'id':entry['id']})

@app.route('/api/feedback',methods=['GET'])
def api_fb_get(): return jsonify(feedback_store[-20:])

@app.route('/ping')
def ping():
    """Keep-alive endpoint — prevents Render free tier spin-down"""
    return jsonify({'ok':True,'trained':engine.is_trained,'ts':datetime.datetime.now().isoformat()})

if __name__=='__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(debug=False,host='0.0.0.0',port=port,threaded=True)
