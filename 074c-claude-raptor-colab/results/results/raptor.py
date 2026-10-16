import json, time, random, math, sys, os
import numpy as np, torch
from collections import defaultdict
T0=time.time()
def log(*a):
    s=f"[{time.time()-T0:7.1f}s] "+" ".join(str(x) for x in a)
    print(s,flush=True); open('/content/progress.log','a').write(s+"\n")
NART=int(os.environ.get('NART',8)); MODEL=os.environ.get('MODEL','Qwen/Qwen3-4B-Instruct-2507')
from datasets import load_dataset
ds=load_dataset('emozilla/quality',split='validation')
log('cols',ds.column_names,len(ds))
arts=defaultdict(list)
for r in ds: arts[r['article']].append(r)
keys=sorted(arts.keys()); random.Random(0).shuffle(keys); keys=keys[:NART]
log('articles',len(keys),'questions',sum(len(arts[k]) for k in keys))
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side='left'
llm=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16).cuda().eval()
emb=SentenceTransformer('BAAI/bge-small-en-v1.5',device='cuda')
def chat(u): return tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True)
@torch.no_grad()
def gen(prompts,maxnew=160,bs=12):
    out=[]
    for i in range(0,len(prompts),bs):
        enc=tok([chat(p) for p in prompts[i:i+bs]],return_tensors='pt',padding=True).to('cuda')
        o=llm.generate(**enc,max_new_tokens=maxnew,do_sample=False)
        out+= [tok.decode(x[enc.input_ids.shape[1]:],skip_special_tokens=True).strip() for x in o]
    return out
L=[tok.encode(c,add_special_tokens=False)[0] for c in "ABCD"]
@torch.no_grad()
def answer(ctx,q,opts):
    o="\n".join(f"{c}. {t}" for c,t in zip("ABCD",opts))
    p=(f"Context:\n{ctx}\n\n" if ctx else "")+f"Answer the multiple-choice question{' about the article using the context' if ctx else ''}.\n\nQuestion: {q}\n{o}\n\nRespond with the option letter only (A, B, C, or D)."
    enc=tok(chat(p),return_tensors='pt').to('cuda')
    lg=llm(**enc).logits[0,-1,L]
    return int(lg.argmax()), enc.input_ids.shape[1]
def ntok(t): return len(tok.encode(t,add_special_tokens=False))
def chunks(a,n=100):
    w=a.split(); return [" ".join(w[i:i+n]) for i in range(0,len(w),n)]
def E(t): return emb.encode(t,normalize_embeddings=True,batch_size=64)
from sklearn.cluster import KMeans
def build(a):
    leaves=chunks(a); nodes=[(t,0) for t in leaves]; cur=leaves; ce=E(cur); calls=0; lvl=0
    while len(cur)>1 and lvl<3:
        lvl+=1
        k=max(1,math.ceil(len(cur)/5))
        if len(cur)<=6 or lvl==3: groups=[list(range(len(cur)))]
        else:
            lab=KMeans(k,n_init=5,random_state=0).fit_predict(ce); groups=[list(np.where(lab==g)[0]) for g in range(k)]; groups=[g for g in groups if g]
        ps=["Write a concise summary (under 100 words) of the following passages, keeping key facts, names and events.\n\n"+"\n\n".join(cur[i] for i in g) for g in groups]
        s=gen(ps,maxnew=150); calls+=len(ps)
        nodes+= [(t,lvl) for t in s]; cur=s; ce=E(cur)
    return nodes,calls
BUDGET=1500
def retrieve(nodes,nemb,q,opts):
    qe=E(q+" "+" ".join(opts)); sc=nemb@qe; ctx=[];used=0;
    for i in np.argsort(-sc):
        n=ntok(nodes[i][0])
        if used+n>BUDGET: continue
        ctx.append(nodes[i][0]); used+=n
        if used>BUDGET-60: break
    return "\n\n".join(ctx)
runs={'config':{'model':MODEL,'n_articles':len(keys),'budget':BUDGET},'arms':{},'preds':[]}
res=defaultdict(lambda:{'c':0,'n':0,'tok':0}); bt=0; ncalls=0; levels=defaultdict(int)
for ai,k in enumerate(keys):
    qs=arts[k]; t=time.time(); nodes,c=build(k); dt=time.time()-t; bt+=dt; ncalls+=c
    for _,l in nodes: levels[l]+=1
    log(f'art {ai} words={len(k.split())} nodes={len(nodes)} summ_calls={c} build={dt:.1f}s qs={len(qs)}')
    leaf=[n for n in nodes if n[1]==0]
    ne=E([n[0] for n in nodes]); le=ne[:len(leaf)]
    for qi,r in enumerate(qs):
        gold=r['answer']; opts=r['options']; q=r['question']; rec={'article':ai,'q':q,'gold':gold}
        ctxs={'zero':"", 'flat':retrieve(leaf,le,q,opts), 'raptor':retrieve(nodes,ne,q,opts), 'full':k}
        for arm,ctx in ctxs.items():
            p,nt=answer(ctx,q,opts); res[arm]['c']+=p==gold; res[arm]['n']+=1; res[arm]['tok']+=ntok(ctx) if ctx else 0
            rec[arm]=p
            if arm=='raptor': rec['raptor_levels']=None
        runs['preds'].append(rec)
    log({a:(v['c']/v['n'],v['n']) for a,v in res.items()})
runs['arms']={a:{'accuracy':v['c']/v['n'],'n':v['n'],'mean_context_tokens':v['tok']/v['n']} for a,v in res.items()}
runs['tree_build']={'seconds':bt,'summary_calls':ncalls,'nodes_by_level':dict(levels)}
json.dump(runs,open('/content/runs.json','w'),indent=1); log('DONE',runs['arms'],runs['tree_build'])
