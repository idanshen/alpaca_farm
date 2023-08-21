import os
import argparse

from best_of_n import run_decode_augmented
from alpaca_farm.utils import jload, jdump
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator, alpaca_leaderboard_general

# Set up argparse to take in the model name and checkpoint dir
parser = argparse.ArgumentParser()
parser.add_argument("--reward_name_or_path", type=str, default="huggyllama/llama-7b"
                    , help="The name or path of the decoder to use")
parser.add_argument('--reward_checkpoint_dir', type=str, default=''
                    , help="The path to the checkpoint directory of the decoder (adapter weigthts)")
parser.add_argument('--path_to_result', type=str, default='./results.json'
                    , help='The path to the output json file')
parser.add_argument('--per_device_batch_size', type=int, default=12
                    , help='The batch size to use for decoding')
parser.add_argument('--load_in_4_bits', type=bool, default=True
                    , help='Whether to load the model in 4 bits')
parser.add_argument('--exp_name', type=str, default='eval_outputs_rm'
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
    list_dict_data1 = jload(args.output_filepath1)
    list_dict_data2 = jload(args.output_filepath2)
else:
    raise Exception('Output file(s) not found!')
    

print("Finish generating data, start evaluating")
alpaca_leaderboard_general([list_dict_data1, list_dict_data2], is_print_metrics=True, annotators_config = "annotator_pool_v0/configs.yaml", name=args.exp_name)# , **decoding_kwargs)
