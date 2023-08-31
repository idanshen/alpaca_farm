import os
import argparse
from typing import List, Dict, Any

import transformers
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass, field

from best_of_n import run_decode_augmented
from alpaca_farm import data_utils, common
from alpaca_farm.utils import jload, jdump
from alpaca_farm.models import reward_model as reward_model_module
from alpaca_farm.rl.ppo_trainer import _make_left_padded_tokenizer

"""
Arguments for the script
"""
@dataclass
class Arguments:
    reward_model_name_or_path: str = field(
        default="huggyllama/llama-7b", metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."})
    reward_model_checkpoint_dir: str = field(
        default="", metadata={"help": "Path to a checkpoint directory."})
    output_filepath: str = field(
        default="./outputs.json", metadata={"help": "Path to a checkpoint directory."})
    path_to_result: str = field(
        default="results.json", metadata={"help": "The path to the output json file."})
    per_device_batch_size: int = field(
        default=12, metadata={"help": "The path to the output json file."})
    four_bits: bool = field(default=True, metadata={"help": "If True, uses 4-bit quantization."})
    flash_attn: bool = field(default=False, metadata={"help": "If True, uses Flash Attention."})
    bf16: bool = field(
        default=False, metadata={"help": "If True, uses bfloat16. If lora and four_bits are True, bfloat16 is used for the lora weights."})
    fp16: bool = field(
        default=False, metadata={"help": "If True, uses float16. "})
    use_lora: bool = field(default=True, metadata={"help": "If True, uses LoRA."})
    transformer_cache_dir: str = field(
        default=None,
        metadata={
            "help": "Path to a directory where transformers will cache the model. "
            "If None, transformers will use the default cache directory."
        },)


def make_reward_model(args, is_trainable=False):
    reward_model_config = reward_model_module.RewardConfig(backbone_model_name_or_path=args.reward_model_name_or_path)
    # for pretrained reward models that aren't lora-based
    if reward_model_config.backbone_model_name_or_path != 'huggyllama/llama-7b':
        base_reward_model = reward_model_module.RewardNoLoraModel(
            transformer_cache_dir=args.transformer_cache_dir,
            four_bits=args.four_bits,
            bfloat16=args.bf16,
            flash_attn=args.flash_attn,
            is_trainable=is_trainable,
            config=reward_model_config, )
    else:
        base_reward_model = reward_model_module.RewardModel(
            transformer_cache_dir=args.transformer_cache_dir,
            four_bits=args.four_bits,
            bfloat16=args.bf16,
            use_lora=args.use_lora,
            flash_attn=args.flash_attn,
            pretrained_lora_weights=args.reward_model_checkpoint_dir,
            is_trainable=is_trainable,
            config=reward_model_config, )
    return base_reward_model


@torch.inference_mode()
def evaluate_data(args, reward_model, eval_data_list_dict) -> List[Dict[str, Any]]:
    """Given a generated dataset, evaluate it using the reward model
    
    args: argparse.Namespace, the arguments to use
    reward_model: reward_model_module.RewardModel, the reward model to use
    eval_data_list_dict: List[Dict[str, Any]], the generated data to evaluate
    """

    reward_model.eval()
    
    pbar = tqdm(total=len(eval_data_list_dict), desc="eval")
    rewards_list = []

    print('Evaluating reward scores...')
    for idx in range(0, len(eval_data_list_dict), args.per_device_batch_size):
        if len(eval_data_list_dict) > (idx + args.per_device_batch_size):
            batch_list_dict = eval_data_list_dict[idx:idx+args.per_device_batch_size]
        else:
            batch_list_dict = eval_data_list_dict[idx:]

        batch_full_outputs = [l['prompt'] + ' ' + l['output'] for l in batch_list_dict]
        encoded_full_responses = reward_tokenizer(batch_full_outputs, return_tensors="pt", padding=True, truncation=True)
        encoded_full_responses, = common.prepare_inputs((encoded_full_responses, ), device=0)
        reward_outputs = reward_model(**encoded_full_responses)
        rewards = reward_outputs['rewards'].squeeze().cpu().detach().numpy().tolist()

        rewards_list.extend(rewards if isinstance(rewards, list) else [rewards])
        # join data
        pbar.update(len(batch_list_dict))
    
    print('Combining reward outputs into outputs...')
    for j in range(len(eval_data_list_dict)):
        eval_data_list_dict[j]['reward'] = rewards_list[j]
        eval_data_list_dict[j]['reward_model'] = args.reward_model_name_or_path + args.reward_model_checkpoint_dir

    print('Finished evaluating reward scores!')
    
    print('Mean reward score: ', sum(rewards_list) / len(rewards_list))
    print('Std reward score: ', torch.tensor(rewards_list).std().item())

    return eval_data_list_dict


if __name__ == "__main__":
    # parse arguments
    parser = transformers.HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()

    if os.path.isfile(args.output_filepath):
        eval_data_list_dict = jload(args.output_filepath)
    else:
        raise Exception('Output file(s) not found!')
    
    print('Loaded data, now evaluating reward scores...')
    # set up tokenizer and set padding and truncation side to the right
    reward_tokenizer: transformers.PreTrainedTokenizer = _make_left_padded_tokenizer(model_name_or_path=args.reward_model_name_or_path)
    reward_tokenizer.padding_side = "right"
    reward_tokenizer.truncation_side = "right"
    
    reward_model = make_reward_model(args=args)

    # mixed precision
    if args.fp16:
        mixed_precision = 'fp16'
    elif args.bf16:
        mixed_precision = 'bf16'
    else:
        mixed_precision = None
    reward_model.forward = common.cast_with_native_amp(reward_model.forward, mixed_precision=mixed_precision)

    eval_data = evaluate_data(args, reward_model, eval_data_list_dict)

    # combine output file and reward outputs

    OUTPUT_DIR = './outputs/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    args.path_to_result = os.path.join(OUTPUT_DIR, args.path_to_result)
    print(f'Saving results to file {args.path_to_result}...')
    jdump(eval_data, args.path_to_result)
