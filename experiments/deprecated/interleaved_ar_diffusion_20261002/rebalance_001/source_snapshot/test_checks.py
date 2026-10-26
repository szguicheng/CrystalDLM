"""One runnable check for the experiment's critical contracts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
assert (ROOT / 'score.py').exists(), 'score.py has not implemented the crystal scoring contract'
from score import audit, edit, answer, grade_answer, grade_edit
from score import parse_fn_wyckoff
import numpy as np

source = (ROOT.parents[1] / 'research/plaid_20260930/train_mp-2657.txt').read_text().strip()
assert audit(source)['valid'], audit(source)
equivalent=source.replace('Ti 0.000 0.000 0.000 2a','Ti 0.500 0.500 0.500 2a')
assert audit(equivalent)['valid'], 'a representative on the same decoded Wyckoff orbit was rejected'
signature=lambda s:sorted((atom.specie.symbol,*((np.round(atom.frac_coords%1,8))%1)) for atom in s)
assert signature(parse_fn_wyckoff(source,to_cif=False))==signature(parse_fn_wyckoff(equivalent,to_cif=False))
assert audit(equivalent)['decoder_max_representative_shift_A']<1e-9
assert not audit(source.replace('Ti 0.000 0.000 0.000 2a','Ti 0.500 0.500 0.250 2a'))['valid']
assert audit(source.replace('0.305', '1.305'))['valid'], 'periodic coordinate aliases must remain valid'
assert not audit(source.replace('Ti2O4', 'H999'))['valid']
assert not audit(source.replace('Sites (6)', 'Sites (999)'))['valid']
assert not audit(source.replace('O 0.305 0.305', 'O 0.300 0.305'))['valid']
changed = edit(source, '1.05')
assert not grade_edit(edit(equivalent,'1.05'),source,'1.05')['success'], 'edit must preserve raw coordinates, even when equivalent'
assert 'abc: 4.88 4.88 3.12' in changed
assert grade_edit(changed, source, '1.05')['success']
assert not grade_edit(source, source, '1.05')['success']
assert not grade_edit(changed.replace('0.305', '0.306'), source, '1.05')['success']
assert grade_answer(answer(source), source)['success']
alternative = 'Space group: 136. Formula: Ti2O4. 6 atoms and 2 orbits. a=4.65, b=4.65, c=2.97.'
assert grade_answer(alternative, source)['success']
assert not grade_answer(alternative.replace('6 atoms', '7 atoms'), source)['success']
assert not grade_answer(alternative + ' Space group: 1.', source)['success']
assert not grade_answer('I cannot determine the structure.', source)['success']
print('PASS: structure, header contradictions, Wyckoff constraint, Decimal edit and factual response scoring')

assert (ROOT / 'prepare.py').exists(), 'prepare.py has not implemented prefix-target tasks'
from prepare import tasks
record = dict(id='mp-2657', c1=source, plus=changed, minus=edit(source, '.95'), condition=dict(formula='Ti2O4', sg=136))
examples = tasks(record, 'train', 'plus')
assert len(examples) == 4
assert [e['task'] for e in examples] == ['G', 'Q1', 'E', 'Q2']
assert '0.305' not in examples[0]['prefix'] and 'mp-2657' not in examples[0]['prefix']
assert source in examples[1]['prefix']
assert source in examples[2]['prefix'] and '4.88' not in examples[2]['prefix']
assert changed in examples[3]['prefix']
print('PASS: material-specific tasks expose only previously completed crystal blocks')

assert (ROOT / 'model.py').exists(), 'model.py has not implemented hybrid attention and absorbing diffusion'
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from model import Hybrid, attention, reverse_step, unpack_block, target_positions

torch.manual_seed(17)
config = Qwen2Config(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.)
config._attn_implementation = 'eager'
net = Hybrid(Qwen2ForCausalLM(config), start_id=30, end_id=31, mask_id=32, pad_id=33).eval()
ids = torch.tensor([1, 2, 30, 5, 6, 7, 8])
mask, regions = attention(ids, 'hybrid', (3, 7), 30, 31)
assert not mask[0, 0, 2, 3] and mask[0, 0, 3, 6]
assert (regions[3:] > 0).all() and len(regions[3:].unique()) == 1
first = net.logits(ids, 'hybrid', (3, 7), 32)
other = ids.clone(); other[6] = 9
second = net.logits(other, 'hybrid', (3, 7), 32)
assert torch.equal(first[:3], second[:3]), 'future crystal leaked into earlier text logits'
assert not torch.equal(first[3], second[3]), 'within-crystal bidirectional attention is absent'
ar1 = net.logits(ids, 'ar', (3, 7), 0)
ar2 = net.logits(other, 'ar', (3, 7), 0)
assert torch.equal(ar1[:6], ar2[:6])
assert target_positions(3, 4, 'ar') == [2, 3, 4, 5]
assert target_positions(3, 4, 'hybrid') == [3, 4, 5, 6]
current = torch.tensor([32, 4, 32, 6])
predicted = torch.tensor([1, 2, 3, 4])
for k in range(32, 0, -1):
 current = reverse_step(current, predicted, k, 32)
assert current.tolist() == [1, 4, 3, 6]
assert unpack_block([4, 5, 31, 33, 33], 31, 33) == [4, 5]
for bad in [[4, 33, 31], [4, 31, 5], [4, 31, 31], [4, 5]]:
 try:
  unpack_block(bad, 31, 33)
 except ValueError:
  pass
 else:
  raise AssertionError(f'invalid block accepted: {bad}')
print('PASS: actual Transformer future isolation, crystal bidirectionality, label positions, absorbing sampler and boundaries')

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(ROOT / 'smoke_data_001/tokenizer')
net.start_id, net.end_id, net.mask_id, net.pad_id = [tok.convert_tokens_to_ids(s) for s in
    ['<CRYSTAL_START>', '<CRYSTAL_END>', '<CRYSTAL_MASK>', '<CRYSTAL_PAD>']]
seed = next(s for s in range(1000) if int(torch.randint(1, 33, (1,), generator=torch.Generator().manual_seed(s))) == 32)
p1 = net.pack(examples[0], tok, 'hybrid', torch.Generator().manual_seed(seed))
different = dict(examples[0], crystal=source.replace('4.65', '1000.00'))
assert len(tok.encode(examples[0]['crystal'])) != len(tok.encode(different['crystal']))
p2 = net.pack(different, tok, 'hybrid', torch.Generator().manual_seed(seed))
assert torch.equal(p1['ids'], p2['ids']) and torch.equal(p1['allowed'], p2['allowed'])
assert torch.equal(p1['times'], p2['times']), 'true target length leaked into full-mask model inputs'
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                 num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.)
cfg._attn_implementation = 'sdpa'
small = Hybrid(Qwen2ForCausalLM(cfg), net.start_id, net.end_id, net.mask_id, net.pad_id)
for mode in ['ar', 'hybrid']:
 small.zero_grad()
 loss, metrics = small.loss(examples[:2], tok, mode, torch.Generator().manual_seed(17))
 assert torch.isfinite(loss) and metrics['text'] is not None and metrics['crystal'] is not None
 loss.backward()
 assert torch.isfinite(small.base.lm_head.weight.grad).all()
import tempfile
with tempfile.TemporaryDirectory() as directory:
 saved = Path(directory) / 'model.pt'
 torch.save(small.state_dict(), saved)
 restored = Hybrid(Qwen2ForCausalLM(cfg), net.start_id, net.end_id, net.mask_id, net.pad_id)
 restored.load_state_dict(torch.load(saved, weights_only=True))
 assert all(torch.equal(a, b) for a, b in zip(small.state_dict().values(), restored.state_dict().values()))
print('PASS: target-length isolation, both objectives backward on SDPA, exact model-state reload')

warm = dict(examples[0], noise_step=32, supervise_lead=False)
packed = small.pack(warm, tok, 'hybrid', torch.Generator().manual_seed(17))
assert [g['kind'] for g in packed['groups']] == ['crystal'], 'crystal-only adaptation still supervises the reference lead'
assert packed['groups'][0]['k'] == 32 and len(packed['groups'][0]['positions']) == 256
assert (packed['ids'][-256:] == small.mask_id).all(), 'forced high-mask training did not produce full MASK'
print('PASS: crystal-only adaptation and explicitly selected full-mask training')

assert (ROOT / 'adapt.py').exists(), 'adapt.py has not implemented disjoint conflict supervision'
from adapt import conflict_example, TRAIN_OFFSETS, TEST_OFFSETS, phase_stream
assert not set(TRAIN_OFFSETS).intersection(TEST_OFFSETS)
old = dict(record, condition=dict(formula='H2', sg=1))
conflict = conflict_example(old, source, 'train')
assert 'H2' in conflict['prefix'] and source in conflict['prefix']
assert grade_answer(conflict['target'], source)['success']
assert 'H2' not in conflict['target']
warm_sequence = phase_stream([record], 'warmup', False, 17)
assert [next(warm_sequence)['task'] for _ in range(2)] == ['G', 'E']
print('PASS: conflicting answers follow the current structure; held-out pairs and crystal-only curriculum are separate')

import diagnose
assert hasattr(diagnose, 'training_model_hash'), 'diagnostic does not understand both actual training receipt schemas'
assert diagnose.training_model_hash({'code_hashes': {'model.py': 'original'}}) == 'original'
assert diagnose.training_model_hash({'source_sha256': {'model.py': 'adaptation'}}) == 'adaptation'
print('PASS: original and adaptation diagnostic provenance schemas')

assert (ROOT / 'rescore.py').exists(), 'explicit a,b,c sentence is not recognized by the audit scorer'
from rescore import reading_grade
plain_axes='Space group: 136. Formula: Ti2O4. 6 atoms and 2 orbits. The a, b, c are 4.65, 4.65, 2.97 angstroms.'
assert not grade_answer(plain_axes, source)['success']
assert reading_grade(plain_axes, source)['success']
assert not reading_grade(plain_axes.replace('2.97', '2.98'), source)['success']
assert not reading_grade(plain_axes+' a=7.00.', source)['success']
print('PASS: unambiguous named-axis prose is scored without relaxing numerical or contradiction checks')

from rescore import score_evaluation
import json
candidate=dict(id=record['id'],swap_id=record['id'],G=False,E_reference=False,flow_complete=False,
               reference_read=dict(ok=True,value=plain_axes),swap_read=dict(ok=False),
               no_crystal_read=dict(ok=False),generation=dict(ok=False))
with tempfile.TemporaryDirectory() as directory:
 folder=Path(directory)
 (folder/'evaluation.jsonl').write_text(json.dumps(candidate)+'\n')
 summary,scored=score_evaluation(folder,{record['id']:record})
 assert summary['n']==1 and summary['counts']['Q_reference']==1
 assert summary['counts']['closed_loop']==0 and summary['counts']['Q_swap']==0
 assert scored[0]['reference_read_grade_v2']['success']
 assert 'reference_read_grade_v2' not in json.loads((folder/'evaluation.jsonl').read_text())
print('PASS: generic cohort rescoring preserves failed attempts and raw output')

assert (ROOT / 'causal_diagnose.py').exists(), 'matched-RNG history diagnostic is not implemented'
from causal_diagnose import canonical_crystal
unusual=source.replace('abc: 4.65 4.65 2.97','abc: 4.650 4.650 2.970').replace('0.305','0.3050')+'\n'
assert canonical_crystal(unusual)==source
from score import parse
precise=source.replace('4.65','4.65123')
assert parse(canonical_crystal(precise))==parse(precise), 'canonicalization rounded a real value'
print('PASS: diagnostic canonicalization is lossless and never substitutes structure values')

import run
assert hasattr(run,'edit_factor'), 'production edit prompts have no shared canonical factor renderer'
assert run.edit_factor('minus')=='0.95' and run.edit_factor('plus')=='1.05'
print('PASS: actual and reference edit requests use identical numeric factor formatting')

assert (ROOT/'align.py').exists(), 'learning-based prompt/history alignment is not implemented'
from align import READ_TRAIN, READ_TEST, augmented_edit, aligned_stream
assert not set(READ_TRAIN).intersection(READ_TEST)
aug=augmented_edit(record,'plus',3)
assert parse(aug['crystal'])==parse(edit(record['c1'],'1.05'))
assert 'mp-2657' not in aug['prefix']
sample=aligned_stream([record,dict(record,id='second')],17)
assert [next(sample)['task'] for _ in range(4)]==['G','Q1','E','Q2']
print('PASS: alignment uses actual-input targets and retains generation anchors without training on new test phrases')

import align
assert hasattr(align,'Hybrid'), 'alignment entrypoint cannot construct its shared model'

assert (ROOT/'cohort.py').exists(), 'unseen-material training stream is not implemented'
from cohort import balanced_stream
s=balanced_stream([record,dict(record,id='second')],17)
counts={}
for _ in range(1024):
 e=next(s)
 if e['task']=='E_varied':
  key=(e['direction'],e['edit_style']);counts[key]=counts.get(key,0)+1
assert all((d,i) in counts for d in ['plus','minus'] for i in range(6)), 'direction/style coverage remains coupled'
print('PASS: training independently covers edit direction and every numeric/history style')

from diagnostics.cohort_probe import select_rows
assert [r['id'] for r in select_rows([record,dict(record,id='second')],['second',record['id']])]==['second',record['id']]
assert len(select_rows([dict(record,id=str(i)) for i in range(20)],[str(i) for i in range(20)]))==16
print('PASS: diagnosis uses frozen material IDs and first 16 selection without output filtering')

from recover import generation_example, generation_stream, recovery_stream
second=dict(record,id='second',c1=changed,plus=edit(changed,'1.05'),minus=edit(changed,'0.95'))
g=generation_example(record,2)
assert g['crystal']==source and '0.305' not in g['prefix'] and record['id'] not in g['prefix']
assert g['prefix'].index('136')<g['prefix'].index('Ti2O4')
seq=generation_stream([record,second],17);exposure={record['id']:0,'second':0}
for _ in range(32):
 e=next(seq);exposure[e['source_id']]+=1
 assert e['task']=='G' and not e['supervise_lead']
assert set(exposure.values())=={16}
seq=recovery_stream([record,second],17);batch=[next(seq) for _ in range(8)]
assert [e['task'] for e in batch]==['G','Q1','E','Q2','Q_varied','Q_varied','E_varied','Q_varied']
assert len({e['prefix'].split('\nAssistant:',1)[0] for e in batch})==1
assert grade_edit(batch[6]['crystal'],batch[6]['prefix'].split('<CRYSTAL_START>')[1].split('<CRYSTAL_END>')[0],run.edit_factor(batch[6]['direction']))['success']
print('PASS: generation curriculum preserves targets and balanced coverage; augmented histories retain actual-input edits')

from diagnostics.order_probe import release_counts, choose_positions
counts=release_counts(17,size=4,steps=4)
assert sum(counts)==4 and release_counts(17,size=4,steps=4)==counts
scores=torch.tensor([.1,.9,.2,.8])
assert choose_positions(scores,2,'confidence',torch.Generator().manual_seed(17)).tolist()==[1,3]
assert choose_positions(scores,2,'left',torch.Generator().manual_seed(17)).tolist()==[0,1]
old=torch.tensor([32,7,32,9]);positions=(old==32).nonzero().flatten()
selected=choose_positions(scores[:2],1,'confidence',torch.Generator().manual_seed(17))
new=old.clone();new[positions[selected]]=torch.tensor([5,6])[selected]
assert torch.equal(old[old!=32],new[old!=32]) and int((new==32).sum())==1
print('PASS: matched release counts, confidence ranking, and released-token immutability')

from rebalance import joint_stream
seq=joint_stream([record,second],17);batch=[next(seq) for _ in range(16)]
assert sum(e['task']=='G' for e in batch)==8
assert sum(e['task'] in ['E','E_varied'] for e in batch)==4
assert sum('target' in e for e in batch)==4
assert all(e.get('supervise_lead',True) for e in batch if e['task']=='G')
assert len({e['prefix'].split('\nAssistant:',1)[0] for e in batch})==1
for e in batch:
 if 'target' in e:
  latest=e['prefix'].rsplit('<CRYSTAL_START>',1)[1].split('<CRYSTAL_END>',1)[0]
  assert grade_answer(e['target'],latest)['success']
assert len({id(e) for e in batch})==16, 'repeated tasks share mutable corruption metadata'
print('PASS: each optimizer batch retains 8G/4E/4Q, triggers, latest-structure answers and independent examples')
