import os
import argparse

from dataclasses import dataclass, field

import transformers
from best_of_n import run_decode_augmented, run_decode
from alpaca_farm.utils import jload, jdump
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator, alpaca_leaderboard


# convert the argparse into a dataclass
@dataclass
class Arguments:
    decoder_name_or_path: str = field(
        default="huggyllama/llama-7b", metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."}),
    decoder_checkpoint_dir: str = field(
        default="./", metadata={"help": "Path to a checkpoint directory of the decoder (adapter weights)."}),
    q_checkpoint_dir: str = field(
        default="./", metadata={"help": "Path to a checkpoint directory of the q model (adapter weights)."}),
    path_to_data: str = field(
        default="./output.json", metadata={"help": "Path to a checkpoint directory."}),
    num_return_sequences: int = field(
        default=1, metadata={"help": "The number of sequences to return from the decoder."}),
    temp: float = field(
        default=0.7, metadata={"help": "The temperature to use for decoding."}),
    per_device_batch_size: int = field(
        default=12, metadata={"help": "The batch size to use for decoding."}),
    load_in_4_bits: bool = field(
        default=True, metadata={"help": "Whether to load the model in 4 bits."}),
    beta: float = field(
        default=1.0, metadata={"help": "The beta value to use for weighting the q model."}),
    exp_name: str = field(default="qmodel", metadata={"help": "The name of the experiment."}),

if __name__ == "__main__":
    # parse arguments
    parser = transformers.HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()


    if 'OPENAI_API_KEY' not in os.environ:
        decoding_kwargs = dict(
            openai_api_key = "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",
            # openai_organization_ids = ["MIT"],
        )
        assert decoding_kwargs["openai_api_key"] is not None, "OPENAI_API_KEY not found you should set it in environment or above"
    else:
        decoding_kwargs = {}

    path_to_data = args.path_to_data

    if os.path.isfile(path_to_data):
        list_dict_data = jload(path_to_data)
    else:
        print('Start generating data')
        if args.q_checkpoint_dir == '':
            print('No q model checkpoint dir is provided, using the default decoder model')

            list_dict_data = run_decode(decoder_name_or_path=args.decoder_name_or_path,
                                        checkpoint_dir=args.decoder_checkpoint_dir,
                                        num_return_sequences=args.num_return_sequences, 
                                        temperature=args.temp, 
                                        per_device_batch_size=args.per_device_batch_size, 
                                        load_in_4_bits=args.load_in_4bits)
        else: 
            list_dict_data = run_decode_augmented(decoder_name_or_path=args.decoder_name_or_path,
                                        checkpoint_dir=args.decoder_checkpoint_dir,
                                        q_checkpoint_dir=args.q_checkpoint_dir,
                                        num_return_sequences=args.num_return_sequences, 
                                        temperature=args.temp, 
                                        per_device_batch_size=args.per_device_batch_size, 
                                        load_in_4_bits=args.load_in_4_bits,
                                        beta=args.beta,)
        print('Saving generated data to {}'.format(path_to_data))
        jdump(list_dict_data, path_to_data)

# print("Finish generating data, start evaluating")
# alpaca_leaderboard(list_dict_data, is_print_metrics=True, annotators_config = "annotator_pool_v0/configs.yaml", name=args.exp_name)# , **decoding_kwargs)

"""
                                        n_draws  n_total  n_wins  n_wins_base  standard_error  win_rate
GPT4                                      17.00   805.00  639.00       149.00            1.38     80.43
ChatGPT                                    9.00   804.00  489.00       306.00            1.71     61.38
rlhf_llama_7b_regen_v7_3ep_v12_ckpt_20     9.00   803.00  370.00       424.00            1.75     46.64
sft_llama_7b_regen_v7_3ep                 16.00   804.00  320.00       468.00            1.72     40.80
my_ppo                                     0.00   781.00  272.00       509.00            1.71     34.83
sft_test_2                                 0.00   787.00  262.00       525.00            1.68     33.29
sft_test_1                                 0.00   796.00  229.00       567.00            1.61     28.77
Davinci001                                 0.00   805.00  201.00       604.00            1.53     24.97
LLaMA 7B                                   0.00   786.00   94.00       692.00            1.16     11.96

"""