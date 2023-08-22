import os
import argparse
from typing import List, Dict, Any

import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from best_of_n import run_decode_augmented
from alpaca_farm import data_utils, common
from alpaca_farm.utils import jload, jdump
from alpaca_farm.models import reward_model as reward_model_module
from alpaca_farm.rl.ppo_trainer import _make_left_padded_tokenizer


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

    generated_data = []
    pbar = tqdm(total=len(eval_data_list_dict), desc="eval")

    for i, idx in enumerate(range(len(eval_data_list_dict)), step=args.per_device_batch_size):
        batch_list_dict = eval_data_list_dict[idx:idx+args.per_device_batch_size]
        
        batch_full_outputs = ...
        encoded_full_responses = reward_tokenizer(batch_full_outputs, return_tensors="pt", padding=True, truncation=True)
        encoded_full_responses, = common.prepare_inputs((encoded_full_responses, ), device=0)
        reward_outputs = reward_model(**encoded_full_responses)
        rewards = reward_outputs['rewards'].cpu().detach().numpy()

        # join data
        pbar.update(args.per_device_batch_size)

        # TODO: maybe do everything here the batching and tokenizing? and joining the output files too actually

        return generated_data

if __name__ == "__main__":
    # Set up argparse to take in the model name and checkpoint dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model_name_or_path", type=str, default="huggyllama/llama-7b"
                        , help="The name or path of the decoder to use")
    parser.add_argument('--reward_checkpoint_dir', type=str, default=''
                        , help="The path to the checkpoint directory of the decoder (adapter weigthts)")
    parser.add_argument('--output_filepath', type=str, default='./outputs.json',
                        help='The path to the output json file to evaluate the samples of')
    parser.add_argument('--path_to_result', type=str, default='./results.json'
                        , help='The path to the output json file')
    parser.add_argument('--per_device_batch_size', type=int, default=12
                        , help='The batch size to use for decoding')
    parser.add_argument('--load_in_4_bits', type=bool, default=True
                        , help='Whether to load the model in 4 bits')
    parser.add_argument('--exp_name', type=str, default='eval_outputs_rm'
                        , help='The name of the experiment')
    args = parser.parse_args()


    if os.path.isfile(args.output_filepath):
        eval_data_list_dict = jload(args.output_filepath)
    else:
        raise Exception('Output file(s) not found!')
    
    print('Loaded data, now evaluating reward scores...')
    reward_tokenizer: transformers.PreTrainedTokenizer = _make_left_padded_tokenizer(model_name_or_path=args.reward_model_name_or_path)
    reward_model = make_reward_model(args=args)
    # eval_data_module: dict = data_utils.make_eval_data_module(tokenizer=reward_tokenizer)
    # eval_datset = eval_data_module["dataset"]
    # data_collator = eval_data_module["data_collator"]
    # print(f"Test dataset size: {len(eval_datset)}")
    # data_loader = DataLoader(
    #     dataset=eval_datset,
    #     collate_fn=data_collator,
    #     batch_size=args.per_device_batch_size,
    #     shuffle=True,
    #     drop_last=True,
    # )
    eval_data = evaluate_data(args, reward_model, eval_data_list_dict)

    # combine output file and reward outputs
    jdump(eval_data, args.output_file)
