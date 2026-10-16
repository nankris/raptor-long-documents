"""Download only public NarrativeQA metadata and public-domain Gutenberg stories."""
import csv,io,json,re,urllib.request,gzip,hashlib,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,default=1);args=parser.parse_args()
BASE='https://raw.githubusercontent.com/google-deepmind/narrativeqa/master/'
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Research-NarrativeQA-074r/1.0'})
    return urllib.request.urlopen(req,timeout=45).read().decode('utf-8-sig',errors='replace')
def csvget(name): return list(csv.DictReader(io.StringIO(get(BASE+name))))
docs=csvget('documents.csv'); qs=csvget('qaps.csv')
by={}
for q in qs: by.setdefault(q['document_id'],[]).append(q)
upper=60000 if args.round==1 else 80000
elig=[d for d in docs if d['set']=='valid' and d['kind']=='gutenberg' and 20000<int(d['story_word_count'])<=upper and len(by[d['document_id']])>=25]
selected=[]; exclusions=[]
if args.round==2:
    with gzip.open(ROOT/'results/data.json.gz','rt') as f: selected=json.load(f)[:6]
    # Identified from source metadata/full-text headers, before any round-2 predictions.
    exclude_titles={'Goblin Market','Der Judenstaat','Cratylus (dialogue)'}
    exclusions=[{'title':t,'reason':'collection or non-narrative work'} for t in sorted(exclude_titles)]
    elig=[d for d in elig if d['wiki_title'] not in exclude_titles and d['document_id'] not in {r['id'] for r in selected}]
for d in sorted(elig,key=lambda d:d['document_id']):
    match=re.search(r'(?:/ebooks/|/files/|/cache/epub/)(\d+)',d['story_url'])
    if not match: raise ValueError(d['story_url'])
    ident=match.group(1)
    url=f'https://www.gutenberg.org/cache/epub/{ident}/pg{ident}.txt'
    try:
        text=get(url)
        start=re.search(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',text,re.I)
        end=re.search(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK',text,re.I)
        if start: text=text[start.end():]
        # End position must be recomputed after trimming start.
        end=re.search(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK',text,re.I)
        if end: text=text[:end.start()]
        words=len(text.split())
        if words<=20000: raise ValueError(f'actual words {words} <=20000')
        rec={'id':d['document_id'],'title':d['wiki_title'],'url':url,'words':words,'sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'questions':[{'q':q['question'],'refs':[q['answer1'],q['answer2']]} for q in by[d['document_id']][:25]]}
        selected.append(rec); print(len(selected),rec['title'],words,flush=True)
        if len(selected)==12: break
    except Exception as e:
        exclusions.append({'id':d['document_id'],'url':url,'error':str(e)});print('exclude',exclusions[-1],flush=True)
assert len(selected)==12, f'only {len(selected)} stories'
with gzip.open(ROOT/'results/data.json.gz','wt') as f: json.dump(selected,f)
json.dump({'source':BASE,'round':args.round,'metadata_upper_words':upper,'eligible':len(elig),'exclusions':exclusions,'selected':[{k:v for k,v in d.items() if k not in ('text','questions')} for d in selected]},open(ROOT/f'results/manifest_round{args.round}.json','w'),indent=2)
