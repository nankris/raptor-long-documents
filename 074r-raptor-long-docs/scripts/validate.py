"""Audit saved outputs, paired sample, source identity and token budgets."""
import gzip, hashlib, json
from pathlib import Path
from scipy.stats import binomtest
root = Path(__file__).resolve().parents[1]
with gzip.open(root/'results/data.json.gz', 'rt') as f:
    sources = json.load(f)
for d in sources:
    assert len(d['text'].split()) == d['words'] > 20000
    assert hashlib.sha256(d['text'].encode()).hexdigest() == d['sha256']
for round_no, n in [(1, 150), (2, 300)]:
    data = json.loads((root/f'results/round{round_no}.json').read_text())
    rows = data['preds']
    assert len(rows) == n
    assert len({(r['document_id'], r['question_index']) for r in rows}) == n
    assert len(data['metadata']) == n//25
    for r in rows:
        src = sources[r['story']]
        assert r['document_id'] == src['id']
        assert r['question'] == src['questions'][r['question_index']]['q']
        assert r['refs'] == src['questions'][r['question_index']]['refs']
        assert set(r['arms']) == {'flat', 'raptor', 'truncated'}
        for a in r['arms'].values():
            assert a['context_tokens'] <= 1500
            assert a['input_tokens'] + 64 <= 2048
            assert a['correct'] == (a['f1'] >= .5)
    wins = sum(r['arms']['raptor']['correct'] and not r['arms']['flat']['correct'] for r in rows)
    losses = sum(r['arms']['flat']['correct'] and not r['arms']['raptor']['correct'] for r in rows)
    p = binomtest(wins, wins+losses, .5).pvalue if wins+losses else 1.
    assert abs(p-data['primary']['p']) < 1e-12
    print(f'Round {round_no}: {n} distinct paired questions; source hashes, budgets and McNemar verified.')
