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
        default='', metadata={"help": "Path to a checkpoint directory of the q model (adapter weights)."}),
    dataset_path: str = field(
        default='', metadata={"help": "Path to a HF dataset."}),
    dataset_name: str = field(
        default='', metadata={"help": "Name of the HF dataset."}),
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


if __name__ == "__main__":
    # parse arguments
    parser = transformers.HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()

    if os.path.isfile(args.path_to_data):
        print('Output file already exists, skipping generating data')
    else:
        print('Start generating data')
        if args.q_checkpoint_dir is None:
            print('No q model checkpoint dir is provided, using the default decoder model')

            list_dict_data = run_decode(decoder_name_or_path=args.decoder_name_or_path,
                                        checkpoint_dir=args.decoder_checkpoint_dir,
                                        dataset_path=args.dataset_path,
                                        dataset_name=args.dataset_name,
                                        num_return_sequences=args.num_return_sequences, 
                                        temperature=args.temp, 
                                        per_device_batch_size=args.per_device_batch_size, 
                                        load_in_4_bits=args.load_in_4bits)
        else: 
            list_dict_data = run_decode_augmented(decoder_name_or_path=args.decoder_name_or_path,
                                        checkpoint_dir=args.decoder_checkpoint_dir,
                                        q_checkpoint_dir=args.q_checkpoint_dir,
                                        dataset_path=args.dataset_path,
                                        dataset_name=args.dataset_name,
                                        num_return_sequences=args.num_return_sequences, 
                                        temperature=args.temp, 
                                        per_device_batch_size=args.per_device_batch_size, 
                                        load_in_4_bits=args.load_in_4_bits,
                                        beta=args.beta,)
        print('Saving generated data to {}'.format(args.path_to_data))
        jdump(list_dict_data, args.path_to_data)