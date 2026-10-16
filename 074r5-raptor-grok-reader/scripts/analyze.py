#!/usr/bin/env python3
import json,collections,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((R/'results').glob('article_*.json')):rows+=json.load(open(p))
if not rows:raise SystemExit('No completed paired articles: no statistical estimate is possible.')
ids=[r['id'] for r in rows];assert len(ids)==len(set(ids))
groups=collections.defaultdict(list)
for r in rows:
 assert r['gold'] in range(4)
 for a in ['flat','raptor']:assert r['arms'][a]['tokens']<=1200 and r['arms'][a]['prediction'] in range(4)
 groups[r['article_id']].append(r)
F=np.array([r['arms']['flat']['prediction']==r['gold'] for r in rows]);T=np.array([r['arms']['raptor']['prediction']==r['gold'] for r in rows]);d=T.astype(int)-F.astype(int)
sums=[];sizes=[]
for aid in sorted(groups):
 rs=groups[aid];sizes.append(len(rs));sums.append(sum(int(r['arms']['raptor']['prediction']==r['gold'])-int(r['arms']['flat']['prediction']==r['gold']) for r in rs))
sums=np.array(sums);sizes=np.array(sizes);rng=np.random.default_rng(74);boot=[]
for i in range(20000):
 idx=rng.integers(0,len(groups),len(groups));boot.append(sums[idx].sum()/sizes[idx].sum())
ci95=np.quantile(boot,[.025,.975]);ci90=np.quantile(boot,[.05,.95])
if ci95[0]>0:verdict='PROVEN';reason='RAPTOR superiority: article-cluster 95% CI is above zero.'
elif ci95[1]<0:verdict='DISPROVEN';reason='RAPTOR inferiority: article-cluster 95% CI is below zero.'
elif ci90[0]>-.05 and ci90[1]<.05:verdict='DISPROVEN';reason='Equivalent within the prespecified ±5 percentage-point margin.'
else:verdict='STILL INCONCLUSIVE';reason='Neither fixed criterion met at the fixed sample/time budget.'
b=int(((T==1)&(F==0)).sum());c=int(((T==0)&(F==1)).sum());p=binomtest(b,b+c,.5).pvalue if b+c else 1.
raws=[]
for fp in (R/'raw').glob('*.json'):
 try:
  raw=json.load(open(fp))
  if 'usage' in raw:raws.append((str(fp.relative_to(R)),raw))
 except Exception:pass
costs=[float(o['total_cost_usd']) for _,o in raws if isinstance(o.get('total_cost_usd'),(float,int))]
models=collections.Counter(k for _,o in raws for k in o.get('modelUsage',{}))
stats=dict(n=len(rows),articles=len(groups),flat_correct=int(F.sum()),raptor_correct=int(T.sum()),flat_accuracy=float(F.mean()),raptor_accuracy=float(T.mean()),difference=float(d.mean()),ci95=ci95.tolist(),ci90=ci90.tolist(),raptor_only=b,flat_only=c,mcnemar_p=p,verdict=verdict,reason=reason,bootstrap_replicates=20000,seed=74,cu=0,gpu=None,grok_cost_usd=sum(costs),calls_with_usage=len(raws),calls_with_cost=len(costs),reported_models=dict(models),summary_fraction=float(np.mean([np.mean(np.array(r['arms']['raptor']['levels'])>0) for r in rows])))
(R/'results/statistics.json').write_text(json.dumps(stats,indent=2))
(R/'results/usage.json').write_text(json.dumps([dict(file=f,usage=o.get('usage'),modelUsage=o.get('modelUsage'),total_cost_usd=o.get('total_cost_usd'),total_cost_usd_ticks=o.get('total_cost_usd_ticks')) for f,o in raws],indent=2))
fig,ax=plt.subplots(figsize=(7,4));v=100*d.mean();ax.errorbar([v],[0],xerr=[[v-100*ci95[0]],[100*ci95[1]-v]],fmt='o',capsize=7,label='95% article-cluster CI');ax.errorbar([v],[0],xerr=[[v-100*ci90[0]],[100*ci90[1]-v]],fmt='o',capsize=4,lw=4,label='90% CI');ax.axvspan(-5,5,color='green',alpha=.09,label='Equivalence ±5 pp');ax.axvline(0,color='gray',ls='--');ax.set_yticks([]);ax.set_xlabel('RAPTOR − flat accuracy (percentage points)');ax.set_title(f'{verdict}: {len(rows)} questions / {len(groups)} articles');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2));fig.tight_layout();fig.savefig(R/'chart.png',dpi=180,bbox_inches='tight')
fmt=lambda a:', '.join(f'{x*100:+.2f}' for x in a)
text=f'''# Round 5: RAPTOR versus flat retrieval

**{verdict}** — {reason}

| Measure | Result |
|---|---:|
| Questions / articles | {len(rows)} / {len(groups)} |
| Flat | {F.sum()}/{len(rows)} ({100*F.mean():.2f}%) |
| RAPTOR | {T.sum()}/{len(rows)} ({100*T.mean():.2f}%) |
| Paired difference | {100*d.mean():+.2f} percentage points |
| Article-cluster 95% CI | [{fmt(ci95)}] pp |
| Article-cluster 90% CI | [{fmt(ci90)}] pp |
| RAPTOR-only / flat-only correct | {b} / {c} |
| Exact two-sided McNemar p (descriptive) | {p:.5g} |
| GPU / CU | None / 0 |
| Grok usage-reported cost | ${sum(costs):.6f} |
| CLI calls with usage / with cost | {len(raws)} / {len(costs)} |

All questions in each selected article are included in SHA256 article order, through the first complete article reaching the 600 target. The first 40 form a parsing-only pilot, retained in the final analysis. Bootstrap resamples articles, retaining all questions, 20,000 replicates, seed 74. No efficacy-based stopping or exclusions. The 1,200-token limit uses cl100k_base, a declared proxy rather than an unavailable native Grok tokenizer. Both arms use identical instructions and per-question evidence ceilings. Batch arm order alternates, with no shared conversation between arms. Summary node fraction among retrieved RAPTOR nodes averages {100*stats['summary_fraction']:.1f}%.

The five-point equivalence margin is an operational threshold for paying tree-construction and summary-inference overhead, not proof of mathematical equality. Cluster inference reflects this sample of articles; batched reader items can also share inference conditions. Public benchmark familiarity or prior training contamination cannot be ruled out. The experiment does not establish behavior for book-length inputs, other readers, other budgets or an optimized RAPTOR implementation.

Raw CLI JSON, prompts, token usage, cached trees, contexts, predictions and attempt events are retained locally. Costs include successful probes and all raw calls reporting usage; failed requests without reported billing cannot be assigned an invented zero dollar cost. No GPU sessions were created. See README for preregistration and model compatibility notes. Run `scripts/resume.sh` to resume missing work and `/workspace/tools/mlenv/bin/python scripts/analyze.py` to regenerate this report.
'''
(R/'writeup.md').write_text(text);print(json.dumps(stats,indent=2))
