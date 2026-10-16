@torch.inference_mode()
def answers(ps):
 out=[]
 for i in range(0,len(ps),4):
  enc=tok([chat(p) for p in ps[i:i+4]],return_tensors='pt',padding=True).to('cuda')
  lg=llm(**enc,logits_to_keep=1).logits[:,-1,letters].float()
  out.extend([{'prediction':int(x),'letter_logits':y} for x,y in zip(lg.argmax(-1).tolist(),lg.cpu().tolist())])
 return out

@torch.inference_mode()
def gen(ps):
 out=[]; i=0; cap=8
 lens=[nt(chat(p)) for p in ps]
 while i<len(ps):
  bs=min(cap,len(ps)-i)
  while bs>1 and bs*max(lens[i:i+bs])>12000:bs=max(1,bs//2)
  try:
   enc=tok([chat(p) for p in ps[i:i+bs]],return_tensors='pt',padding=True).to('cuda')
   z=llm.generate(**enc,max_new_tokens=150,do_sample=False,pad_token_id=tok.eos_token_id)
   out.extend(tok.batch_decode(z[:,enc.input_ids.shape[1]:],skip_special_tokens=True))
   i+=bs;del enc,z
  except torch.OutOfMemoryError:
   if bs==1:raise
   cap=max(1,bs//2)
   if "enc" in locals():del enc
   import gc
   gc.collect();torch.cuda.empty_cache()
 return out
meta=json.load(open(ROOT+'/metadata.json'))
meta.update({'model_commit':llm.config._commit_hash,'tokenizer_commit':tok.init_kwargs.get('_commit_hash'),'transformers':__import__('transformers').__version__,'embedding_model':'BAAI/bge-small-en-v1.5'})
json.dump(meta,open(ROOT+'/metadata.json','w'))
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

step()
