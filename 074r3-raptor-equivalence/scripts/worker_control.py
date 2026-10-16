import pathlib,subprocess,time,json,tarfile,sys
R=pathlib.Path('/workspace/work/tig/074r3-raptor-equivalence');res=R/'results'
end=time.monotonic()+9600
log=open(res/'worker_transport.log','a',buffering=1)
def call(args):
 try:
  p=subprocess.run(['colab',*args],capture_output=True,text=True,timeout=45)
  log.write(time.strftime('%FT%TZ',time.gmtime())+' '+str(args[:3])+' '+str(p.returncode)+'\n')
  return p.returncode==0
 except subprocess.TimeoutExpired:
  log.write('TRANSPORT_TIMEOUT\n');return False
n=json.loads((res/'checkpoint/progress.json').read_text())['completed_articles'];acked=-1
try:
 while time.monotonic()<end:
  if not call(['download','-s','r3074','/content/batch_ready.json',str(res/'ready.json')]):time.sleep(5);continue
  ready=json.loads((res/'ready.json').read_text())
  if 'error' in ready:raise RuntimeError(ready['error'])
  k=ready['completed_articles']
  if k>n:
   if not call(['download','-s','r3074','/content/checkpoint.tar.gz',str(res/'checkpoint.next.tar.gz')]):time.sleep(5);continue
   with tarfile.open(res/'checkpoint.next.tar.gz') as tf:
    inside=json.load(tf.extractfile('checkpoint/progress.json'))
    assert inside['completed_articles']==k
    tf.extractall(res,filter='data')
   (res/'checkpoint.next.tar.gz').replace(res/'checkpoint.tar.gz');n=k
   print(f'SAVED {n}/115',flush=True)
  if acked<n:
   (res/'ack.json').write_text(json.dumps({'completed_articles':n}))
   if call(['upload','-s','r3074',str(res/'ack.json'),'/content/batch_ack.json']):acked=n
  if n==115:print('ALL_DONE',flush=True);break
  time.sleep(5)
 else:print('BUDGET_DEADLINE',flush=True)
finally:
 p=subprocess.run(['colab','stop','-s','r3074'],capture_output=True,text=True,timeout=60)
 (res/'stop.log').open('a').write(p.stdout+p.stderr)
 (res/'stopped_at.txt').write_text(time.strftime('%FT%TZ',time.gmtime()))
