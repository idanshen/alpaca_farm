import os
import argparse
from typing import List, Dict, Any

import transformers
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
        default="huggyllama/llama-7b", metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."}
    ),
    reward_model_checkpoint_dir: str = field(
        default="./", metadata={"help": "Path to a checkpoint directory."}
    ),
    output_filepath: str = field(
        default="./outputs.json", metadata={"help": "Path to a checkpoint directory."}
    ),
    path_to_result: str = field(
        default="./results.json", metadata={"help": "The path to the output json file."}
    ),
    per_device_batch_size: int = field(
        default=12, metadata={"help": "The path to the output json file."}
    ),
    load_in_4_bits: bool = field(
        default=True, metadata={"help": "Whether to load the model in 4 bits."}
    ),
    four_bits: bool = field(default=True, metadata={"help": "If True, uses 4-bit quantization."})
    bfloat16: bool = field(
        default=False, metadata={"help": "If True, uses bfloat16 quantization. If lora and four_bits are True, bfloat16 is used for the lora weights."})
    use_lora: bool = field(default=True, metadata={"help": "If True, uses LoRA."})
    exp_name: str = field(default="eval_outputs_rm", metadata={"help": "The name of the experiment."}),


def make_reward_model(args, is_trainable=False):
    reward_model_config = reward_model_module.RewardConfig(backbone_model_name_or_path=args.reward_model_name_or_path)
    # for pretrained reward models that aren't lora-based
    if reward_model_config.backbone_model_name_or_path != 'huggyllama/llama-7b':
        base_reward_model = reward_model_module.RewardNoLoraModel(
            transformer_cache_dir=args.transformer_cache_dir,
            four_bits=args.four_bits,
            bfloat16=args.bfloat16,
            flash_attn=args.flash_attn,
            is_trainable=is_trainable,
            config=reward_model_config, )
    else:
        base_reward_model = reward_model_module.RewardModel(
            transformer_cache_dir=args.transformer_cache_dir,
            four_bits=args.four_bits,
            bfloat16=args.bfloat16,
            use_lora=args.use_lora,
            flash_attn=args.flash_attn,
            pretrained_lora_weights=args.reward_model_checkpoint_dir,
            is_trainable=is_trainable,
            config=reward_model_config, )
    return base_reward_model


def evaluate_data(args, reward_model, eval_data_list_dict) -> List[Dict[str, Any]]:
    """Given a generated dataset, evaluate it using the reward model
    
    args: argparse.Namespace, the arguments to use
    reward_model: reward_model_module.RewardModel, the reward model to use
    eval_data_list_dict: List[Dict[str, Any]], the generated data to evaluate
    """

    pbar = tqdm(total=len(eval_data_list_dict), desc="eval")
    rewards_list = []

    print('Evaluating reward scores...')
    for i, idx in enumerate(range(len(eval_data_list_dict)), step=args.per_device_batch_size):
        if len(eval_data_list_dict) < (idx + args.per_device_batch_size):
            batch_list_dict = eval_data_list_dict[idx:idx+args.per_device_batch_size]
        else:
            batch_list_dict = eval_data_list_dict[idx:]

        batch_full_outputs = [l['prompt'] + ' ' + l['output'] for l in batch_list_dict]
        encoded_full_responses = reward_tokenizer(batch_full_outputs, return_tensors="pt", padding=True, truncation=True)
        encoded_full_responses, = common.prepare_inputs((encoded_full_responses, ), device=0)
        reward_outputs = reward_model(**encoded_full_responses)
        rewards = reward_outputs['rewards'].squeeze().cpu().detach().numpy()
        
        rewards_list.extend(list(rewards))
        # join data
        pbar.update(len(batch_list_dict))
    
    print('Combining reward outputs into outputs...')
    
    for i in range(len(eval_data_list_dict)):
        eval_data_list_dict[i]['reward'] = rewards_list[i]

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
    reward_tokenizer: transformers.PreTrainedTokenizer = _make_left_padded_tokenizer(model_name_or_path=args.reward_model_name_or_path)
    reward_model = make_reward_model(args=args)

    eval_data = evaluate_data(args, reward_model, eval_data_list_dict)

    # combine output file and reward outputs
    print(f'Saving results to file {args.path_to_result}...')
    jdump(eval_data, args.path_to_result)
