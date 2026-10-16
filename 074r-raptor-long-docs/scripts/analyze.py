"""Preregistered paired analysis and reproducible report/chart generation."""
from pathlib import Path
import json,math
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
ARMS=['flat','raptor','truncated']
def wilson(k,n):
    z=1.95996398454;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-half,mid+half]
def paired(rows,a,b):
    x=np.array([int(r['arms'][a]['correct'])-int(r['arms'][b]['correct']) for r in rows])
    w=int(sum(x==1));l=int(sum(x==-1));n=len(x)
    p=float(binomtest(w,w+l,.5).pvalue) if w+l else 1.
    rng=np.random.default_rng(74)
    boot=x[rng.integers(n,size=(10000,n))].mean(axis=1)
    # Equal questions per story, resample entire story means.
    stories=sorted(set(r['story'] for r in rows))
    means=np.array([np.mean([x[i] for i,r in enumerate(rows) if r['story']==s]) for s in stories])
    cb=means[rng.integers(len(stories),size=(10000,len(stories)))].mean(axis=1)
    return {'difference':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'cluster_ci95':np.quantile(cb,[.025,.975]).tolist(),'wins':w,'losses':l,'p':p}
def pct(x): return f'{100*x:.1f}%'
def interval(ci): return f'[{100*ci[0]:.1f}, {100*ci[1]:.1f}]'
results=[]
for file in sorted((ROOT/'results').glob('round[12].json')):
    data=json.load(open(file));rows=data['preds'];n=len(rows)
    arms={}
    for a in ARMS:
        k=sum(r['arms'][a]['correct'] for r in rows)
        arms[a]={'correct':k,'n':n,'accuracy':k/n,'ci95':wilson(k,n),'mean_f1':float(np.mean([r['arms'][a]['f1'] for r in rows])),'mean_context_tokens':float(np.mean([r['arms'][a]['context_tokens'] for r in rows])),'max_context_tokens':max(r['arms'][a]['context_tokens'] for r in rows)}
        assert arms[a]['max_context_tokens']<=1500
    primary=paired(rows,'raptor','flat')
    verdict=('PROVEN' if primary['difference']>0 else 'DISPROVEN') if primary['p']<.025 else 'STILL INCONCLUSIVE'
    assert data['primary']['verdict']==verdict
    r={'round':data['config']['round'],'n':n,'stories':len(data['metadata']),'arms':arms,'primary':primary,'secondary_raptor_vs_truncated':paired(rows,'raptor','truncated'),'verdict':verdict}
    results.append(r)
assert results
json.dump(results,open(ROOT/'results/analysis.json','w'),indent=2)
last=results[-1]
text=['# 074r: RAPTOR on full NarrativeQA documents','',f'**Verdict: {last["verdict"]}.** Primary claim: RAPTOR improves binary QA accuracy over flat retrieval under a 1,500-token evidence budget.','',
'Qwen3-4B-Instruct-2507 reads and summarizes; bge-small embeds; 100-word leaves and up to three KMeans summary levels reproduce the original retrieval approach. Unlike the original short-article test, every source document exceeds 20,000 words. The usable reader context is deliberately limited to 2,048 tokens. These documents do **not** exceed Qwen’s advertised native context; conclusions apply to the configured budget.','',
'NarrativeQA is free-answer QA. Correctness means maximum normalized token F1 against either reference ≥0.5. This is an automated lexical proxy, not human-judged semantic correctness. It can miss paraphrases or count partially correct answers. Native free-answer generation replaces the original multiple-choice letter scoring.','',
'Summaries preserve the original prompt, 150-token generation limit, and approximate groups of five. Oversized KMeans clusters and the final level are partitioned to fit the same context limit without silently dropping source passages. All arms use identical answer prompts, greedy decoding, a 64-token answer cap, and ≤1,500 evidence tokens including separators. Retrieval uses the question only; there are no answer options or gold references in retrieval.','']
for r in results:
    text.extend([f'## Round {r["round"]}: {r["n"]} questions, {r["stories"]} documents','', '| Arm | Correct | Accuracy (95% Wilson CI) | Mean token F1 | Mean evidence tokens |','|---|---:|---:|---:|---:|'])
    for a in ARMS:
        v=r['arms'][a];text.append(f'| {a} | {v["correct"]}/{v["n"]} | {pct(v["accuracy"])} {interval(v["ci95"])} | {v["mean_f1"]:.3f} | {v["mean_context_tokens"]:.1f} |')
    p=r['primary'];s=r['secondary_raptor_vs_truncated']
    text.extend(['',f'RAPTOR − flat: **{100*p["difference"]:+.1f} percentage points**, paired bootstrap 95% CI {interval(p["ci95"])} points; story-cluster CI {interval(p["cluster_ci95"])} points. Discordant pairs: RAPTOR-only {p["wins"]}, flat-only {p["losses"]}. Exact two-sided McNemar **p={p["p"]:.6g}**; per-look α=0.025. Verdict: **{r["verdict"]}**.', '',f'Secondary RAPTOR − truncated: {100*s["difference"]:+.1f} points, paired 95% CI {interval(s["ci95"])}; McNemar p={s["p"]:.6g} (descriptive).',''])
    raw=json.load(open(ROOT/'results'/f'round{r["round"]}.json'))
    summary_q=sum(any(level>0 for level in row['raptor_levels']) for row in raw['preds'])
    text.extend([f'RAPTOR selected summary nodes for {summary_q}/{r["n"]} questions. Tree construction: {sum(m["summary_calls"] for m in raw["metadata"])} summary calls, {sum(m["build_seconds"] for m in raw["metadata"]):.1f} seconds (one-time, cached across rounds).',''])
    for m in raw['metadata']:
        text.append(f'- {m["title"]}: {m["words"]:,} words, {m["tokens"]:,} Qwen tokens.')
    text.append('')
if len(results)==2:
    text.extend(['Round 1 did not meet the decision threshold. The single redesign doubled the sample from 150 to 300 questions and from 6 to 12 documents, retaining earlier summary trees. A stricter minimal-answer instruction addresses verbosity in lexical grading; all 300 questions receive fresh predictions in every arm, rather than mixing outputs from different prompts. Round 2 is the final planned look; no further sampling or threshold changes were made.',''])
text.extend(['## Interpretation and limitations','',
'Failure to reject McNemar is not evidence of equivalence. Questions within documents are dependent: the question-level exact test is the specified decision rule, while the story-cluster interval shows sensitivity to that dependence. Only 6–12 documents are represented. The split, length restriction, lexical grading, corpus edition differences, and a small local summarizer limit generalization. A public pretrained model may also know these public-domain books. No zero-context contamination control was preregistered.','',
'Gutenberg source files are the complete linked documents rather than Wikipedia summaries. Some linked works may include front matter or additional works; see the sampling audit below. Length alone cannot establish that each question requires evidence from across the whole document.','',
'## Reproduction and artifacts','',
'Run `scripts/prepare.py`, provision a Colab L4, install `transformers==4.57.6 sentence-transformers scikit-learn scipy accelerate`, upload `results/data.json.gz` and `scripts/run.py` to `/content/074r/`, and run `python /content/074r/run.py --round 1`. Analyze the first look before running `--round 2` if needed. Download round JSON and run `python scripts/analyze.py`. Run `scripts/prepare.py --round 2` before the second look to apply the documented sampling redesign; upload the new data file and run.py. Trees from round 1 are reused, but answer predictions are generated afresh using the stricter format instruction. `scripts/run_round1.py` preserves the first execution script. The manifest records public source URLs and content hashes. Raw predictions, references, lengths, software/model versions and summary nodes are retained.','',
'Dataset: [DeepMind NarrativeQA](https://github.com/google-deepmind/narrativeqa). Source texts: public Gutenberg links in `results/manifest.json`. Full context is truncated to the beginning, as requested; it is not a best-window or oracle baseline.',''])
# Runtime/account details and sampling deviations are authored separately, preserved verbatim.
extra=ROOT/'results/notes.md'
if extra.exists():text.extend([extra.read_text()])
(ROOT/'writeup.md').write_text('\n'.join(text))
fig,ax=plt.subplots(figsize=(8,4.8));x=np.arange(3);width=.33
for j,r in enumerate(results):
    vals=np.array([r['arms'][a]['accuracy'] for a in ARMS]);cis=np.array([r['arms'][a]['ci95'] for a in ARMS])
    off=(j-(len(results)-1)/2)*width
    bars=ax.bar(x+off,100*vals,width,yerr=100*np.array([vals-cis[:,0],cis[:,1]-vals]),capsize=4,label=f'Round {r["round"]}, n={r["n"]}')
    ax.bar_label(bars,labels=[f'{100*v:.1f}%' for v in vals],padding=6,fontsize=9)
ax.set_xticks(x,['Flat chunks','RAPTOR tree','Truncated source']);ax.set_ylabel('Binary QA accuracy (%)');ax.set_ylim(0,min(100,max(r['arms'][a]['ci95'][1]*100 for r in results for a in ARMS)+18));ax.legend();ax.spines[['top','right']].set_visible(False)
p=last['primary'];ax.set_title(f'Full NarrativeQA documents >20,000 words\nRAPTOR vs flat: exact McNemar p={p["p"]:.3g}; {last["verdict"]}',fontsize=11)
fig.text(.5,.015,'1,500-token evidence budget · token F1 ≥ 0.5 · error bars: 95% Wilson CI',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(ROOT/'chart.png',dpi=180)
print(json.dumps(results,indent=2))
