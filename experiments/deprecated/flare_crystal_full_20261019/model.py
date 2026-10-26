"""Keep the verified pilot G objective; restore the frozen Q reading task."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = ROOT / 'experiments/flare_crystal_qwen3_8b_20261019/model.py'
spec = spec_from_file_location('verified_flare_crystal_objective', PILOT_PATH)
PILOT = module_from_spec(spec)
spec.loader.exec_module(PILOT)
BLOCK_SIZE = PILOT.BLOCK_SIZE
denoise_logits = PILOT.denoise_logits


def objective_tasks(model, tok, examples, generator, arm):
    assert len(examples) == 2 and examples[0]['task'] == 'G' and examples[1]['task'] == 'Q'
    generation, stats = PILOT.objective(model, tok, examples[0], generator, arm)
    reading, _ = model.loss([examples[1]], tok, 'ar', generator)
    return (generation + reading) / 2, dict(
        clean_ar=(stats['clean_ar'] + float(reading.detach())) / 2,
        denoise=stats['denoise'] / 2, reading_ar=float(reading.detach()))
