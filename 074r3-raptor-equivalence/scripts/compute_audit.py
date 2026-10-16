"""Sanitize local lifecycle evidence; estimate allocation cost, not exact billing."""
import json,pathlib,datetime
R=pathlib.Path(__file__).resolve().parents[1]
p=pathlib.Path.home()/'.config/colab-cli/history/r3074.jsonl'
events=[];allocations=[];start=None
for line in p.read_text().splitlines():
 r=json.loads(line)
 if r['event_type'] not in ('session_created','session_terminated'):continue
 e={k:r[k] for k in ('timestamp','event_type','accelerator','reason') if k in r};events.append(e)
 dt=datetime.datetime.fromisoformat(r['timestamp'])
 if r['event_type']=='session_created':start=dt
 elif start:
  sec=(dt-start).total_seconds();allocations.append({'start':start.isoformat(),'end':dt.isoformat(),'seconds':sec,'estimated_cu':sec*1.54/3600,'termination':r.get('reason')});start=None
out={'events':events,'allocations':allocations,'rate_cu_per_hour':1.54,'total_allocation_seconds':sum(a['seconds'] for a in allocations),'estimated_cu':sum(a['estimated_cu'] for a in allocations),'exact_billing_available':False,'caveat':'Pruned timestamp is local stale-session detection; actual remote termination/billing time is unknown. This is an allocation-window estimate, potentially an overestimate. No further allocation authorized under the exhausted budget.'}
(R/'results/compute_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
