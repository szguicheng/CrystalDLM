"""CPU-only framework labels, frozen data, and adapter-contract acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from .configuration import protocol
from .tokenization import make_processor, encode_row


def audit(dataset_dir):
    import torch
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    p = protocol()
    manifest = dataset_dir / 'manifest.json'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != p['manifest_sha256']:
        raise ValueError('Frozen dataset manifest changed; review new version before proceeding')
    registered = json.loads((dataset_dir / 'dataset_info.json').read_text())
    processor = make_processor('ar_understanding')
    result = {'manifest_sha256': digest, 'scope': 'native LLaMA-Factory CPU preprocessing; no training', 'splits': {}}
    for split, count in [('training', 36362), ('validation', 4501), ('test', 4602)]:
        path = dataset_dir / registered[f'crystal_understanding_{split}']['file_name']
        lengths = []
        sample = None
        for line in path.read_text().splitlines():
            row = json.loads(line)
            item = encode_row(row, processor, 'ar_understanding')
            lengths.append(len(item['input_ids']))
            if sample is None:
                sample = item
                target = [v for v in item['labels'] if v != -100]
                if processor.tokenizer.decode(target) != row['output'] + '<|im_end|>\n':
                    raise ValueError('Supervised labels contain unexpected tokens')
        if len(lengths) != count:
            raise ValueError(f'{split}: expected {count}, got {len(lengths)}')
        collator = SFTDataCollatorWith4DAttentionMask(tokenizer=processor.tokenizer,
            template=processor.template, pad_to_multiple_of=8, label_pad_token_id=-100)
        features = [{k:v for k,v in sample.items() if k != 'segment_ids'}]
        batch = collator(features)
        assert torch.equal(batch['labels'][0, :len(sample['labels'])], torch.tensor(sample['labels']))
        assert (batch['labels'][0, len(sample['labels']):] == -100).all()
        result['splits'][split] = {'rows': count, 'min': min(lengths), 'median': statistics.median(lengths),
            'max': max(lengths), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'non_ignored_labels_decoded': processor.tokenizer.decode(batch['labels'][0][batch['labels'][0] != -100].tolist()),
            'prompt_and_padding_masked': True}
    design_file = dataset_dir / 'examples/design-mp-1210092.json'
    design = encode_row(json.loads(design_file.read_text()), make_processor('ar_design'), 'ar_design')
    result['design_example'] = {'tokens':len(design['input_ids']),
        'description_and_boundary_tokens':design['segment_ids'].count(1), 'crystal_body_and_end_tokens':design['segment_ids'].count(2),
        'full_design_data_audited':False}
    return result


def adapter_check():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from llamafactory.hparams import ModelArguments, FinetuningArguments
    from llamafactory.model.adapter import init_adapter
    torch.manual_seed(42)
    config=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=False)
    base=Qwen3ForCausalLM(config)
    args=ModelArguments(model_name_or_path=protocol()['model_path'],resize_vocab=True)
    finetuning=FinetuningArguments(finetuning_type='lora',lora_rank=4,lora_alpha=8,
        lora_target=','.join(protocol()['lora']['target_modules']),trainable_token_indices=[28,29,30,31])
    model=init_adapter(config,base,args,finetuning,True)
    trainable={name:list(param.shape) for name,param in model.named_parameters() if param.requires_grad}
    assert any('trainable_tokens_delta' in name and shape==[4,32] for name,shape in trainable.items())
    assert not any(shape==[32,32] and 'embed_tokens' in name for name,shape in trainable.items())
    assert any('lm_head' in name and 'lora_B' in name for name in trainable)
    assert all('lora_' in name or 'trainable_tokens_delta' in name for name in trainable)
    ids=torch.tensor([[28,2,29,3,30,4,31]])
    model(input_ids=ids,labels=ids).loss.backward()
    assert any(param.grad is not None and param.grad.abs().sum()>0 for name,param in model.named_parameters() if 'trainable_tokens_delta' in name)
    return {'tiny_model_only':True,'four_input_rows_and_lora_only':True,'trainable_parameters':trainable,'backward_verified':True}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result=audit(args.dataset_dir)
    result['adapter_check']=adapter_check()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='adapter_check'},indent=2))
