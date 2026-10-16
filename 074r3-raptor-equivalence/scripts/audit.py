import json,pathlib,collections,hashlib
R=pathlib.Path(__file__).resolve().parents[1];P=R/'results/checkpoint'
rows=[json.loads(l) for l in (P/'predictions.jsonl').read_text().splitlines()]
keys=[(r['article_id'],r['question_index']) for r in rows]
assert len(keys)==len(set(keys)),'duplicate predictions'
assert all(r['gold'] in range(4) for r in rows)
assert all(r['arms'][a]['prediction'] in range(4) and 0<r['arms'][a]['tokens']<=1500 for r in rows for a in ['flat','raptor'])
assert all(set(r['arms']['flat']['levels'])=={0} for r in rows)
articles={r['article_id']:r for r in rows}
for aid in articles:
 tree=json.loads((P/'trees'/f'{aid}.json').read_text())
 assert all(l in range(4) for _,l in tree)
 assert any(l>0 for _,l in tree)
meta=json.loads((P/'metadata.json').read_text())
out={'questions':len(rows),'articles':len(articles),'planned_questions':meta['questions'],'planned_articles':meta['articles'],'all_complete':len(rows)==meta['questions'] and len(articles)==meta['articles'],'article_words_range':[min(r['words'] for r in rows),max(r['words'] for r in rows)],'article_tokens_range':[min(r['article_tokens'] for r in rows),max(r['article_tokens'] for r in rows)],'summary_retrieved_questions':sum(any(l>0 for l in r['arms']['raptor']['levels']) for r in rows),'mean_evidence_tokens':{a:sum(r['arms'][a]['tokens'] for r in rows)/len(rows) for a in ['flat','raptor']},'identical_context_pairs':sum(r['arms']['flat']['context']==r['arms']['raptor']['context'] for r in rows),'predictions_sha256':hashlib.sha256((P/'predictions.jsonl').read_bytes()).hexdigest()}
(R/'results/audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
