"""One runnable check for the experiment's critical contracts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
assert (ROOT / 'score.py').exists(), 'score.py has not implemented the crystal scoring contract'
from score import audit, edit, answer, grade_answer, grade_edit

source = (ROOT.parents[1] / 'research/plaid_20260930/train_mp-2657.txt').read_text().strip()
assert audit(source)['valid'], audit(source)
assert audit(source.replace('0.305', '1.305'))['valid'], 'periodic coordinate aliases must remain valid'
assert not audit(source.replace('Ti2O4', 'H999'))['valid']
assert not audit(source.replace('Sites (6)', 'Sites (999)'))['valid']
assert not audit(source.replace('O 0.305 0.305', 'O 0.300 0.305'))['valid']
changed = edit(source, '1.05')
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
different = dict(examples[0], crystal=source.replace('4.65', '100.00'))
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
