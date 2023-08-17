import os
import argparse

from best_of_n import run_decode_augmented
from alpaca_farm.utils import jload, jdump
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator, alpaca_leaderboard_general

# Set up argparse to take in the model name and checkpoint dir
parser = argparse.ArgumentParser()
parser.add_argument("--decoder_name_or_path", type=str, default="huggyllama/llama-7b"
                    , help="The name or path of the decoder to use")
parser.add_argument('--decoder_checkpoint_dir', type=str, default=''
                    , help="The path to the checkpoint directory of the decoder (adapter weigthts)")
parser.add_argument('--q_checkpoint_dir', type=str, default=''
                    , help="The path to the checkpoint directory of the q model (adapter weights)")
parser.add_argument('--path_to_result', type=str, default='./results.json'
                    , help='The path to the output json file')
parser.add_argument('--num_return_sequences', type=int, default=1
                    , help='The number of sequences to return from the decoder')
parser.add_argument('--temp', type=float, default=0.7
                    , help='The temperature to use for decoding')
parser.add_argument('--per_device_batch_size', type=int, default=12
                    , help='The batch size to use for decoding')
parser.add_argument('--load_in_4_bits', type=bool, default=True
                    , help='Whether to load the model in 4 bits')
parser.add_argument('--beta', type=float, default=1.0
                    , help='The beta value to use for weighting the q model')
parser.add_argument('--exp_name', type=str, default='qmodel'
                    , help='The name of the experiment')
args = parser.parse_args()

if 'OPENAI_API_KEY' not in os.environ:
    decoding_kwargs = dict(
        openai_api_key = "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",
        # openai_organization_ids = ["MIT"],
    )
    assert decoding_kwargs["openai_api_key"] is not None, "OPENAI_API_KEY not found you should set it in environment or above"
else:
    decoding_kwargs = {}

if os.path.isfile(args.output_filepath1) and os.path.isfile(args.output_filepath2):

else:
    raise Exception('Output file(s) not found!')
    

print("Finish generating data, start evaluating")
alpaca_leaderboard_general[(list_dict_data, ], is_print_metrics=True, annotators_config = "annotator_pool_v0/configs.yaml", name=args.exp_name)# , **decoding_kwargs)

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
