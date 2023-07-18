import os

from dataclasses import dataclass, field

import transformers

from best_of_n import run_decode
from alpaca_farm.utils import jload, jdump

@dataclass
class Arguments:
    model_name_or_path: str = field(
        default="decapoda-research/llama-7b-hf", metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."}
    ),
    checkpoint_dir: str = field(
        default="/data/pulkitag/models/idanshen/alpaca_farm/spf/test_3/adapter_model/", metadata={"help": "Path to a checkpoint directory."}
    ),

if __name__ == "__main__":
    parser = transformers.HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()
    path_to_data = "/data/pulkitag/models/idanshen/alpaca_farm/spf/test_3/responses/output3.json"
    if os.path.isfile(path_to_data):
        list_dict_data = jload(path_to_data)
    else:
        list_dict_data = run_decode(decoder_name_or_path=args.model_name_or_path,
                                    checkpoint_dir=args.checkpoint_dir,
                                    num_return_sequences=1, temperature=1.0, per_device_batch_size=12, load_in_4_bits=True,
                                    dataset_path="tatsu-lab/alpaca_farm",dataset_name="alpaca_instructions", split="unlabeled")
        jdump(list_dict_data, path_to_data)

    print("Finish generating data")