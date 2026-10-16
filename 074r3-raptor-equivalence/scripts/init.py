import os,sys,subprocess,time,json,hashlib,math,random,tarfile
subprocess.check_call([sys.executable,'-m','pip','-q','install','transformers==4.57.6','datasets','sentence-transformers','scikit-learn','accelerate'])
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import numpy as np,torch
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict
os.makedirs('/content/r3074/trees',exist_ok=True)
ROOT='/content/r3074'
MODEL='Qwen/Qwen3-8B'
t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side='left'
llm=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
emb=SentenceTransformer('BAAI/bge-small-en-v1.5',device='cuda')
ds=load_dataset('emozilla/quality',split='validation')
arts=defaultdict(list)
for r in ds: arts[r['article']].append(r)
keys=sorted(arts,key=lambda a:hashlib.sha256(a.encode()).hexdigest())
json.dump({'model':MODEL,'gpu':torch.cuda.get_device_name(),'questions':len(ds),'articles':len(keys),'dataset_fingerprint':ds._fingerprint,'torch':torch.__version__,'start':t0},open(ROOT+'/metadata.json','w'))
def chat(p): return tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def nt(t): return len(tok.encode(t,add_special_tokens=False))
def E(t): return emb.encode(t,normalize_embeddings=True,batch_size=64)
@torch.inference_mode()
def gen(ps):
 out=[]
 for i in range(0,len(ps),2):
  enc=tok([chat(p) for p in ps[i:i+2]],return_tensors='pt',padding=True).to('cuda')
  z=llm.generate(**enc,max_new_tokens=150,do_sample=False,pad_token_id=tok.eos_token_id)
  out.extend(tok.batch_decode(z[:,enc.input_ids.shape[1]:],skip_special_tokens=True))
 return out
letters=[tok.encode(x,add_special_tokens=False)[0] for x in 'ABCD']
@torch.inference_mode()
def answers(ps):
 out=[]
 for i in range(0,len(ps),2):
  enc=tok([chat(p) for p in ps[i:i+2]],return_tensors='pt',padding=True).to('cuda')
  lg=llm(**enc,logits_to_keep=1).logits[:,-1,letters].float()
  out.extend([{'prediction':int(x),'letter_logits':y} for x,y in zip(lg.argmax(-1).tolist(),lg.cpu().tolist())])
 return out

def build(a):
 w=a.split(); cur=[' '.join(w[i:i+100]) for i in range(0,len(w),100)]; nodes=[(t,0) for t in cur]
 for lvl in range(1,4):
  if len(cur)<=1:break
  if len(cur)<=6 or lvl==3:groups=[list(range(len(cur)))]
  else:
   lab=KMeans(max(1,math.ceil(len(cur)/5)),n_init=5,random_state=0).fit_predict(E(cur));groups=[np.where(lab==x)[0].tolist() for x in sorted(set(lab))]
  parts=[]
  for g in groups:
   part=[]
   for j in g:
    if part and nt('\n\n'.join(cur[x] for x in part+[j]))>3000:parts.append(part);part=[]
    part.append(j)
   if part:parts.append(part)
  cur=gen(['Write a concise summary (under 100 words) of the following passages, keeping key facts, names and events.\n\n'+'\n\n'.join(cur[i] for i in g) for g in parts]);nodes.extend((x,lvl) for x in cur)
 return nodes

def retrieve(nodes,ne,qe):
 parts=[];lev=[]
 for i in np.argsort(-(ne@qe)):
  cand='\n\n'.join(parts+[nodes[i][0]])
  if nt(cand)>1500:continue
  parts.append(nodes[i][0]);lev.append(nodes[i][1])
  if nt(cand)>1440:break
 ctx='\n\n'.join(parts);assert nt(ctx)<=1500
 return ctx,lev

def step():
 done=set()
 if os.path.exists(ROOT+'/predictions.jsonl'):
  for l in open(ROOT+'/predictions.jsonl'):done.add(json.loads(l)['article_id'])
 for ai,a in enumerate(keys):
  aid=hashlib.sha256(a.encode()).hexdigest()
  if aid in done:continue
  start=time.time();p=ROOT+'/trees/'+aid+'.json'
  if os.path.exists(p):nodes=json.load(open(p))
  else:nodes=build(a);json.dump(nodes,open(p,'w'))
  ep=ROOT+'/trees/'+aid+'.npy'
  if os.path.exists(ep):ne=np.load(ep)
  else:ne=E([n[0] for n in nodes]);np.save(ep,ne)
  nl=sum(n[1]==0 for n in nodes)
  qs=arts[a]; ps=[];records=[]
  for qi,r in enumerate(qs):
   qe=E(r['question']+' '+' '.join(r['options']))
   rec={'article_id':aid,'article_index':ai,'question_index':qi,'question':r['question'],'options':r['options'],'gold':r['answer'],'words':len(a.split()),'article_tokens':nt(a),'arms':{}}
   for arm,nn,ee in [('flat',nodes[:nl],ne[:nl]),('raptor',nodes,ne)]:
    ctx,lev=retrieve(nn,ee,qe)
    opts='\n'.join(c+'. '+o for c,o in zip('ABCD',r['options']))
    ps.append(f'Context:\n{ctx}\n\nAnswer the multiple-choice question about the article using the context.\n\nQuestion: {r["question"]}\n{opts}\n\nRespond with the option letter only (A, B, C, or D).')
    rec['arms'][arm]={'context':ctx,'tokens':nt(ctx),'levels':lev}
   records.append(rec)
  pred=answers(ps)
  for j,rec in enumerate(records):
   for k,arm in enumerate(['flat','raptor']):rec['arms'][arm].update(pred[2*j+k])
  old=open(ROOT+'/predictions.jsonl').read() if os.path.exists(ROOT+'/predictions.jsonl') else ''
  with open(ROOT+'/predictions.tmp','w') as f:
   f.write(old)
   for rec in records:f.write(json.dumps(rec)+'\n')
   f.flush();os.fsync(f.fileno())
  os.replace(ROOT+'/predictions.tmp',ROOT+'/predictions.jsonl')
  json.dump({'completed_articles':len(done)+1,'total_articles':len(keys),'last_seconds':time.time()-start,'elapsed':time.time()-t0},open(ROOT+'/progress.json','w'))
  with tarfile.open('/content/checkpoint.tar.gz','w:gz') as tf:tf.add(ROOT,arcname='checkpoint')
  print(open(ROOT+'/progress.json').read(),flush=True)
  return
 print('ALL_DONE',flush=True)
print('INITIALIZED',len(ds),len(keys),flush=True)
