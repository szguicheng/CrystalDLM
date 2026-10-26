"""Use the installed LLaMA-Factory processor for labels and segment boundaries."""
from .configuration import protocol


def make_processor(task):
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    p = protocol()
    model_args = ModelArguments(model_name_or_path=p['model_path'],
        model_revision=p['model_revision'], add_special_tokens=','.join(p['tokens']), resize_vocab=True)
    data_args = DataArguments(template=p['template'], enable_thinking=False,
        train_on_prompt=False, cutoff_len=p['tasks'][task]['cutoff_len'], packing=False)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    actual = [tokenizer.convert_tokens_to_ids(x) for x in p['tokens']]
    if actual != p['lora']['trainable_token_indices']:
        raise ValueError(f'Control token identity changed: {actual}')
    processor = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer,
        processor=tokenizer_module['processor'], data_args=data_args)
    return processor


def encode_row(row, processor, task):
    prompt = row['instruction'] + ('\n' + row['input'] if row['input'] else '')
    messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': row['output']}]
    source, target = processor.template.encode_oneturn(processor.tokenizer, messages)
    if len(source) + len(target) > processor.data_args.cutoff_len:
        raise ValueError(f'Sample requires {len(source) + len(target)} tokens; truncation is prohibited')
    ids, labels = processor._encode_data_example(messages[:1], messages[1:], None, None, [], [], [])
    if ids != source + target or labels != [-100] * len(source) + target:
        raise ValueError('Native processor label contract differs from answer-only template')
    segments = [0] * len(source)
    if task == 'ar_understanding':
        segments += [3] * len(target)
    else:
        start, end = protocol()['lora']['trainable_token_indices'][:2]
        if target.count(start) != 1 or target.count(end) != 1:
            raise ValueError('Design reference requires exactly one crystal span')
        left, right = target.index(start), target.index(end)
        if left >= right:
            raise ValueError('Invalid crystal span')
        # START is the AR switch; body and END are the diffusion targets.
        segments += [1] * (left + 1) + [2] * (right - left) + [1] * (len(target) - right - 1)
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels, 'segment_ids': segments}
