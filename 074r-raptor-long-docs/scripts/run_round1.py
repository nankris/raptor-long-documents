import os,json,gzip,time,math,re,string,collections,argparse,hashlib
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from scipy.stats import binomtest
P=argparse.ArgumentParser(); P.add_argument('--round',type=int,required=True); args=P.parse_args()
ROOT=Path('/content/074r'); ROOT.mkdir(exist_ok=True)
START=time.time(); torch.manual_seed(74); np.random.seed(74); torch.set_num_threads(4)
def log(*a):
    s=f'[{time.time()-START:.1f}s] '+' '.join(map(str,a)); print(s,flush=True)
    with open(ROOT/'progress.log','a') as f: f.write(s+'\n')
MODEL='Qwen/Qwen3-4B-Instruct-2507'; BUDGET=1500; USABLE=2048
log('round',args.round,'GPU',torch.cuda.get_device_name(0))
tok=AutoTokenizer.from_pretrained(MODEL);tok.padding_side='left'
llm=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
emb=SentenceTransformer('BAAI/bge-small-en-v1.5',device='cuda')
def ntok(t): return len(tok.encode(t,add_special_tokens=False))
def chat(t): return tok.apply_chat_template([{'role':'user','content':t}],tokenize=False,add_generation_prompt=True)
@torch.inference_mode()
def gen(prompts,maxnew,bs=4):
    out=[]
    for i in range(0,len(prompts),bs):
        texts=[chat(p) for p in prompts[i:i+bs]]
        enc=tok(texts,return_tensors='pt',padding=True).to('cuda')
        assert enc.input_ids.shape[1]+maxnew<=USABLE, ('context cap',enc.input_ids.shape,maxnew)
        o=llm.generate(**enc,max_new_tokens=maxnew,do_sample=False,pad_token_id=tok.pad_token_id)
        out.extend(tok.batch_decode(o[:,enc.input_ids.shape[1]:],skip_special_tokens=True))
    return [s.strip() for s in out]
def E(t): return emb.encode(t,normalize_embeddings=True,batch_size=64,show_progress_bar=False)
PREFIX='Write a concise summary (under 100 words) of the following passages, keeping key facts, names and events.\n\n'
def summary_prompt(ts): return PREFIX+'\n\n'.join(ts)
def splitgroups(cur,groups):
    # Bound summary input rather than silently truncate an oversized cluster.
    bounded=[]
    for g in groups:
        part=[]
        for i in g:
            if part and ntok(chat(summary_prompt([cur[j] for j in part+[i]])))+150>USABLE:
                bounded.append(part); part=[]
            part.append(i)
        if part: bounded.append(part)
    return bounded

def build(text):
    words=text.split(); leaves=[' '.join(words[i:i+100]) for i in range(0,len(words),100)]
    nodes=[{'text':s,'level':0} for s in leaves]; cur=leaves; calls=0
    for level in range(1,4):
        if len(cur)<=1: break
        k=max(1,math.ceil(len(cur)/5))
        if len(cur)<=6 or level==3: groups=[list(range(len(cur)))]
        else:
            labels=KMeans(k,n_init=5,random_state=0).fit_predict(E(cur))
            groups=[list(np.where(labels==g)[0]) for g in range(k)]; groups=[g for g in groups if g]
        groups=splitgroups(cur,groups)
        summaries=gen([summary_prompt([cur[i] for i in g]) for g in groups],150)
        nodes.extend({'text':s,'level':level} for s in summaries)
        calls+=len(summaries);cur=summaries
        log('level',level,'summaries',len(summaries))
    return nodes,calls

def retrieve(nodes,vectors,q):
    ranked=np.argsort(-(vectors@E(q))); selected=[];levels=[]
    for i in ranked:
        candidate='\n\n'.join(selected+[nodes[i]['text']])
        if ntok(candidate)>BUDGET: continue
        selected.append(nodes[i]['text']); levels.append(nodes[i]['level'])
        if ntok(candidate)>BUDGET-60: break
    return '\n\n'.join(selected),levels

def normalize(s):
    s=s.lower().translate(str.maketrans('','',string.punctuation))
    return re.sub(r'\b(a|an|the)\b',' ',s).split()
def f1(a,b):
    a,b=normalize(a),normalize(b)
    if not a or not b: return float(a==b)
    common=sum((collections.Counter(a)&collections.Counter(b)).values())
    return 2*common/(len(a)+len(b))

with gzip.open(ROOT/'data.json.gz','rt') as f: data=json.load(f)
N=6 if args.round==1 else 12
records=[]; metadata=[]
for ai,d in enumerate(data[:N]):
    cache=ROOT/f"story_{ai}.json"
    if cache.exists():
        r=json.load(open(cache));records.extend(r['preds']);metadata.append(r['meta']);log('reused',ai);continue
    text=d['text'];ntext=ntok(text);assert d['words']>20000 and ntext>USABLE
    log('story',ai,d['title'],d['words'],'words',ntext,'tokens')
    t=time.time();nodes,calls=build(text);buildsec=time.time()-t
    vectors=E([n['text'] for n in nodes]); leafn=sum(n['level']==0 for n in nodes)
    prefix=tok.decode(tok.encode(text,add_special_tokens=False)[:BUDGET],skip_special_tokens=True)
    while ntok(prefix)>BUDGET: prefix=tok.decode(tok.encode(prefix,add_special_tokens=False)[:-1])
    jobs=[]; preds=[]
    for qi,r in enumerate(d['questions']):
        ctxs={};levels={}
        for arm,ns,vs in [('flat',nodes[:leafn],vectors[:leafn]),('raptor',nodes,vectors)]:
            ctxs[arm],levels[arm]=retrieve(ns,vs,r['q'])
        ctxs['truncated']=prefix
        rec={'story':ai,'document_id':d['id'],'question_index':qi,'question':r['q'],'refs':r['refs'],'arms':{},'raptor_levels':levels['raptor']}
        preds.append(rec)
        for arm,ctx in ctxs.items():
            assert ntok(ctx)<=BUDGET
            prompt=f'Context:\n{ctx}\n\nAnswer the question about the story using the context. Give only a short answer, without explanation.\nQuestion: {r["q"]}\nAnswer:'
            jobs.append((qi,arm,ntok(ctx),prompt))
    outputs=gen([j[3] for j in jobs],64,bs=6)
    for (qi,arm,ct,prompt),answer in zip(jobs,outputs):
        score=max(f1(answer,ref) for ref in preds[qi]['refs'])
        preds[qi]['arms'][arm]={'prediction':answer,'f1':score,'correct':score>=.5,'context_tokens':ct,'input_tokens':ntok(chat(prompt))}
    meta={'story':ai,'id':d['id'],'title':d['title'],'words':d['words'],'tokens':ntext,'build_seconds':buildsec,'summary_calls':calls,'node_levels':dict(collections.Counter(n['level'] for n in nodes)),'total_seconds':time.time()-t}
    json.dump({'meta':meta,'preds':preds},open(cache,'w'),indent=2)
    with gzip.open(ROOT/f'nodes_{ai}.json.gz','wt') as f: json.dump(nodes,f)
    records.extend(preds);metadata.append(meta)
    log('finished',ai,'seconds',time.time()-t,'scores',{a:sum(r['arms'][a]['correct'] for r in preds) for a in ['flat','raptor','truncated']})
assert len(records)==N*25
wins=sum(r['arms']['raptor']['correct'] and not r['arms']['flat']['correct'] for r in records)
loss=sum(r['arms']['flat']['correct'] and not r['arms']['raptor']['correct'] for r in records)
p=binomtest(wins,wins+loss,.5).pvalue if wins+loss else 1.
verdict=('PROVEN' if wins>loss else 'DISPROVEN') if p<.025 else 'STILL INCONCLUSIVE'
import transformers,sentence_transformers,sklearn
result={'config':{'round':args.round,'model':MODEL,'model_revision':llm.config._commit_hash,'embedding':'BAAI/bge-small-en-v1.5','budget':BUDGET,'usable_context':USABLE,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':transformers.__version__,'sentence_transformers':sentence_transformers.__version__,'sklearn':sklearn.__version__,'seconds':time.time()-START},'metadata':metadata,'preds':records,'primary':{'raptor_only':wins,'flat_only':loss,'p':p,'verdict':verdict}}
json.dump(result,open(ROOT/f'round{args.round}.json','w'),indent=2)
log('DONE',result['primary'])
