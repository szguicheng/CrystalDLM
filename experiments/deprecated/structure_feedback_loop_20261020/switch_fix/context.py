"""Give the existing model a fresh G request with failed output as input data."""


def build_context(item, previous, tok, loop):
    request = item['prompt'].removeprefix('User: ').removesuffix('\nAssistant:')
    has_crystal = bool(previous['crystal_raw_token_ids'])
    prior_ids = previous['crystal_raw_token_ids'] if has_crystal else previous['lead_token_ids']
    body = tok.decode(prior_ids, skip_special_tokens=True)
    feedback = loop._checker_feedback(previous, item['space_group_number'])
    label = 'crystal' if has_crystal else 'response'
    prompt = ('User: ' + request + f'\nPrevious failed {label} (input data):\n' + body +
              '\nChecker feedback:\n' + feedback + '\nAssistant:')
    return tok.encode(prompt, add_special_tokens=False), feedback
