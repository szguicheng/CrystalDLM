"""Test history markers and single-user serialization on the same diagnostic cases."""
import hashlib
import json
from pathlib import Path
import diagnose as D


def main():
    loop = D.LOOP
    trainer = loop._load_module('context_probe_trainer', loop.TRAINER_PATH)
    train, model, tok, identity, source_hash, software = trainer.load_model()
    trainer.load_checkpoint(model, loop.E.DEFAULT_CHECKPOINTS['flare'], train, 'flare')
    model.eval()
    state = train.trainable_sha256(model)
    cases = D.select_cases()
    out = D.HERE / 'context_probe'
    loop.RUNNER._write_json(out / 'manifest.json', dict(cases=[c['case_id'] for c in cases],
        state_sha256=state, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for variant in ('markerless_history', 'single_user_plain'):
        for case in cases:
            replay, previous = case['replay'], case['previous']
            if variant == 'markerless_history':
                ids = [i for i in replay['context_token_ids'] if i not in (model.start_id, model.end_id)]
            else:
                request = case['item']['prompt'].removeprefix('User: ').removesuffix('\nAssistant:')
                body = tok.decode(previous['crystal_raw_token_ids'], skip_special_tokens=True)
                text = ('User: ' + request + '\nPrevious failed crystal (input data):\n' + body +
                        '\nChecker feedback:\n' + replay['feedback'] + '\nAssistant:')
                ids = tok.encode(text, add_special_tokens=False)
            result = D.sample(ids, 256, model, tok, replay['seed'])
            result.update(case_id=case['case_id'],variant=variant,context_token_ids=ids,
                          seed=replay['seed'],state_sha256=state)
            loop.RUNNER._write_json(out / variant / (case['case_id'] + '.json'),result)
        rows=[json.loads(q.read_text()) for q in (out / variant).glob('*.json')]
        print(json.dumps(dict(variant=variant,n=len(rows),start=sum(r['stop']=='start' for r in rows),
                              eos=sum(r['stop']=='eos' for r in rows),limit=sum(r['stop']=='limit' for r in rows))),flush=True)
    assert train.trainable_sha256(model) == state


if __name__ == '__main__':main()
