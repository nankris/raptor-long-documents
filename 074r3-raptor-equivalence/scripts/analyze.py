import json,pathlib
import numpy as np
from scipy.stats import binomtest,norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1]; P=R/'results/checkpoint'
rows=[json.loads(l) for l in (P/'predictions.jsonl').read_text().splitlines()]
n=len(rows); ids=sorted({r['article_id'] for r in rows}); m=len(ids)
f=np.array([r['arms']['flat']['prediction']==r['gold'] for r in rows]);t=np.array([r['arms']['raptor']['prediction']==r['gold'] for r in rows]);d=t.astype(float)-f
rng=np.random.default_rng(74); B=20000
qb=np.concatenate([d[rng.integers(n,size=(min(500,B-i),n))].mean(1) for i in range(0,B,500)])
sums=np.array([sum(d[i] for i,r in enumerate(rows) if r['article_id']==a) for a in ids]);ns=np.array([sum(r['article_id']==a for r in rows) for a in ids]);ix=rng.integers(m,size=(B,m));cb=sums[ix].sum(1)/ns[ix].sum(1)
def ci(x,alpha):return (100*np.quantile(x,[alpha/2,1-alpha/2])).tolist()
q95,c95=ci(qb,.05),ci(cb,.05);q90,c90=ci(qb,.1),ci(cb,.1)
if min(q95[0],c95[0])>0:verdict='PROVEN';reason='superiority'
elif max(q95[1],c95[1])<0:verdict='DISPROVEN';reason='inferiority'
elif min(q90[0],c90[0])>-3 and max(q90[1],c90[1])<3:verdict='DISPROVEN';reason='equivalent within ±3 points'
elif max(q90[1],c90[1])<3:verdict='DISPROVEN';reason='no practically meaningful advantage (one-sided 95% upper bounds <3 points)'
else:verdict='STILL INCONCLUSIVE';reason='fixed compute/time budget exhausted before either endpoint established'
a=int(sum(t&~f));b=int(sum(f&~t));p=binomtest(a,a+b).pvalue if a+b else 1
out={'n':n,'documents':m,'flat_correct':int(f.sum()),'raptor_correct':int(t.sum()),'flat_accuracy':float(f.mean()),'raptor_accuracy':float(t.mean()),'difference_pp':float(d.mean()*100),'question_95':q95,'cluster_95':c95,'question_90':q90,'cluster_90':c90,'raptor_only':a,'flat_only':b,'mcnemar_p':p,'verdict':verdict,'reason':reason}
(R/'results/statistics.json').write_text(json.dumps(out,indent=2))
fig,ax=plt.subplots(figsize=(8,4.8))
for y,c,label in [(1,q95,'Question bootstrap 95%'),(0,c95,'Article-cluster bootstrap 95%')]:
 ax.errorbar(d.mean()*100,y,xerr=[[d.mean()*100-c[0]],[c[1]-d.mean()*100]],fmt='o',capsize=6,label=label)
ax.axvspan(-3,3,color='green',alpha=.1,label='Practical equivalence margin');ax.axvline(0,color='black',lw=.8);ax.axvline(3,color='green',ls='--');ax.axvline(-3,color='green',ls='--');ax.set_yticks([0,1],['Article clusters','Questions']);ax.set_xlabel('RAPTOR − flat accuracy (percentage points)');ax.set_title(f'QuALITY: {n} questions / {m} articles\n{verdict}: {reason}');ax.legend(loc='best',fontsize=8);fig.tight_layout();fig.savefig(R/'chart.png',dpi=180)
print(json.dumps(out,indent=2))
