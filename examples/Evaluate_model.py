import os

from examples.best_of_n import run_decode

if 'OPENAI_API_KEY' not in os.environ:
    decoding_kwargs = dict(
        openai_api_key = "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",
        # openai_organization_ids = ["MIT"],
    )
    assert decoding_kwargs["openai_api_key"] is not None, "OPENAI_API_KEY not found you should set it in environment or above"
else:
    decoding_kwargs = {}

from alpaca_farm.utils import jload, jdump
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator, alpaca_leaderboard

path_to_data = "/home/idanshen/projects/alpaca_farm/tmp_ppo/ppo_trial_1/output.json"
if os.path.isfile(path_to_data):
    list_dict_data = jload(path_to_data)
else:
    list_dict_data = run_decode(decoder_name_or_path="decapoda-research/llama-7b-hf",
                                checkpoint_dir="/home/idanshen/projects/alpaca_farm/tmp/test_3/adapter_model/",
                                num_return_sequences=1, temperature=0.7, per_device_batch_size=12, load_in_4_bits=True)
    jdump(list_dict_data, path_to_data)

print("Finish generating data, start evaluating")
alpaca_leaderboard(list_dict_data, is_print_metrics=True, annotators_config = "annotator_pool_v0/configs.yaml", name="my_ppo")# , **decoding_kwargs)

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