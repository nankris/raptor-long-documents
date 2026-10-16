import json,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];events=[]
for l in (r/'attempts.jsonl').read_text().splitlines():
 try:events.append(json.loads(l))
 except Exception:pass
files=list((r/'results').glob('article_*.json'));n=sum(len(json.load(open(p))) for p in files)
raw=[]
for p in (r/'raw').glob('*.json'):
 try:
  x=json.load(open(p))
  if 'usage' in x:raw.append(x)
 except Exception:pass
print(json.dumps(dict(completed_articles=len(files),completed_questions=n,pilot=(r/'results/pilot.json').exists(),usage_calls=len(raw),usage_cost_usd=sum(x.get('total_cost_usd',0) for x in raw),last_event=events[-1] if events else None,deadline_remaining_minutes=round((json.load(open(r/'start.json'))['deadline']-time.time())/60,1))))
