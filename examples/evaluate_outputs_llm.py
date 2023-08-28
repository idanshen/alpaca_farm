import os
import argparse

import transformers
from dataclasses import dataclass, field

from alpaca_farm.utils import jload, jdump
from alpaca_farm.auto_annotations import alpaca_leaderboard_general

# arguments
@dataclass
class Arguments:
    output_filepath1: str = field(
        default="./outputs1.json", metadata={"help": "Path to outputs from a model (reference)."}),
    output_filepath2: str = field(
        default="./outputs2.json", metadata={"help": "Path to outputs from a model to compare (new)."}),
    path_to_result: str = field(
        default="results.json", metadata={"help": "The path to the output json file."}),
    exp_name: str = field(default="eval_outputs_llm", metadata={"help": "The name of the experiment."}),

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

    if os.path.isfile(args.output_filepath1) and os.path.isfile(args.output_filepath2):
        list_dict_data1 = jload(args.output_filepath1)
        list_dict_data2 = jload(args.output_filepath2)
    else:
        raise Exception('Output file(s) not found!')
        

    print("Finished loading outputs, start evaluating") # TODO (seungwook): fix config files
    alpaca_leaderboard_general([list_dict_data1, list_dict_data2], is_print_metrics=True, annotators_config = "annotator_pool_v0/configs.yaml", name=args.exp_name)# , **decoding_kwargs)

    """
                                            n_draws  n_total  n_wins  n_wins_base  standard_error  win_rate
    GPT4                                      17.00   805.00  639.00       149.00            1.38     80.43
    ChatGPT                                    9.00   804.00  489.00       306.00            1.71     61.38
    rlhf_llama_7b_regen_v7_3ep_v12_ckpt_20     9.00   803.00  370.00       424.00            1.75     46.64
    sft_llama_7b_regen_v7_3ep                 16.00   804.00  320.00       468.00            1.72     40.80
    my_ppo                                     0.00   781.00  272.00       509.00            1.71     34.83
    sft_test_5                                23.00   801.00  282.00       496.00            1.68     36.64
    sft_test_2                                 0.00   787.00  262.00       525.00            1.68     33.29
    sft_test_1                                 0.00   796.00  229.00       567.00            1.61     28.77
    Davinci001                                 0.00   805.00  201.00       604.00            1.53     24.97
    LLaMA 7B                                   0.00   786.00   94.00       692.00            1.16     11.96

    """
