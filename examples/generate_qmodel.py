import os

from dataclasses import dataclass, field

import transformers

from best_of_n import run_decode_augmented, run_decode
from alpaca_farm.utils import jdump
from alpaca_farm import accelerate_patch

# convert the argparse into a dataclass
@dataclass
class Arguments:
    decoder_name_or_path: str = field(
        default="huggyllama/llama-7b", metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."})
    decoder_checkpoint_dir: str = field(
        default="./", metadata={"help": "Path to a checkpoint directory of the decoder (adapter weights)."})
    sft_checkpoint_dir: str = field(
        default='', metadata={"help": "Path to a checkpoint directory of the sft model (adapter weights) -- to use for calculating KL for PPO."})
    q_checkpoint_dir: str = field(
        default='', metadata={"help": "Path to a checkpoint directory of the q model (adapter weights)."})
    num_q_heads: int = field(
        default=1, metadata={"help": "The number of q heads to use for decoding."})
    q_head_type: str = field(
        default='linear', metadata={"help": "The type of q head to use for decoding."})
    dataset_path: str = field(
        default='', metadata={"help": "Path to a HF dataset."})
    dataset_name: str = field(
        default='', metadata={"help": "Name of the HF dataset."})
    path_to_result: str = field(
        default="output.json", metadata={"help": "Path to a output/result file to be saved."})
    num_return_sequences: int = field(
        default=1, metadata={"help": "The number of sequences to return from the decoder."})
    temp: float = field(
        default=0.7, metadata={"help": "The temperature to use for decoding."})
    greedy: bool = field(
        default=False, metadata={"help": "If True, uses greedy decoding."})
    per_device_batch_size: int = field(
        default=12, metadata={"help": "The batch size to use for decoding."})
    load_in_4_bits: bool = field(
        default=True, metadata={"help": "Whether to load the model in 4 bits."})
    flash_attn: bool = field(
        default=False, metadata={"help": "If True, uses Flash Attention."})
    beta: float = field(
        default=1.0, metadata={"help": "The beta value to use for weighting the q model."})
    bf16: bool = field(
        default=False, metadata={"help": "If True, uses bfloat16. If lora and four_bits are True, bfloat16 is used for the lora weights."})
    fp16: bool = field(
        default=False, metadata={"help": "If True, uses float16. "})

if __name__ == "__main__":
    # parse arguments
    parser = transformers.HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()

    # sanity checking for flash attn b/c bs needs to be 1
    if args.flash_attn:
        assert args.per_device_batch_size == 1, "Flash attn needs batch size of 1"

    # mixed precision
    if args.fp16:
        mixed_precision = 'fp16'
    elif args.bf16:
        mixed_precision = 'bf16'
    else:
        mixed_precision = None

    accelerator = accelerate_patch.MyAccelerator(
        mixed_precision=mixed_precision,
        log_with=[],
    )

    decoding_kwargs = {}
    
    if args.greedy:
        args.temp = 1.0
        decoding_kwargs = {
            'do_sample': False,
            'num_beams': 1,
            'top_p': 1.0,
            'top_k': 1,
        }

    if os.path.isfile(args.path_to_result):
        print('Output file already exists, skipping generating data')
    else:
        print('Start generating data')
        # ppo or sft
        if args.q_checkpoint_dir == '' and args.sft_checkpoint_dir == '': #and no multiple lora checkpoints
            print('No q model checkpoint dir is provided, using the default decoder model')

            list_dict_data = run_decode(decoder_name_or_path=args.decoder_name_or_path,
                                        checkpoint_dir=args.decoder_checkpoint_dir,
                                        dataset_path=args.dataset_path,
                                        dataset_name=args.dataset_name,
                                        num_return_sequences=args.num_return_sequences, 
                                        temperature=args.temp, 
                                        per_device_batch_size=args.per_device_batch_size, 
                                        load_in_4_bits=args.load_in_4_bits,
                                        flash_attn=args.flash_attn,
                                        accelerator=accelerator,
                                        **decoding_kwargs)
            avg_kl = None
        # ppo, where sft is provided for kl calculation
        elif args.q_checkpoint_dir == '' and args.sft_checkpoint_dir != '':
            list_dict_data, avg_kl = run_decode_augmented(decoder_name_or_path=args.decoder_name_or_path,
                                        checkpoint_dir=args.decoder_checkpoint_dir,
                                        sft_checkpoint_dir=args.sft_checkpoint_dir,
                                        dataset_path=args.dataset_path,
                                        dataset_name=args.dataset_name,
                                        num_return_sequences=args.num_return_sequences, 
                                        temperature=args.temp, 
                                        per_device_batch_size=args.per_device_batch_size, 
                                        load_in_4_bits=args.load_in_4_bits,
                                        flash_attn=args.flash_attn,
                                        accelerator=accelerator,
                                        beta=args.beta,
                                        **decoding_kwargs)
            args.path_to_result = 'kl_{}_'.format(avg_kl) + args.path_to_result if avg_kl is not None else args.path_to_result 
        # q model
        elif args.q_checkpoint_dir != '' and args.sft_checkpoint_dir == '':
            list_dict_data, avg_kl = run_decode_augmented(decoder_name_or_path=args.decoder_name_or_path,
                                        checkpoint_dir=args.decoder_checkpoint_dir,
                                        q_checkpoint_dir=args.q_checkpoint_dir,
                                        dataset_path=args.dataset_path,
                                        dataset_name=args.dataset_name,
                                        num_return_sequences=args.num_return_sequences, 
                                        temperature=args.temp, 
                                        per_device_batch_size=args.per_device_batch_size, 
                                        load_in_4_bits=args.load_in_4_bits,
                                        flash_attn=args.flash_attn,
                                        accelerator=accelerator,
                                        beta=args.beta,
                                        num_q_heads=args.num_q_heads,
                                        q_head_type=args.q_head_type,
                                        **decoding_kwargs)
            args.path_to_result = 'kl_{}_'.format(avg_kl) + args.path_to_result if avg_kl is not None else args.path_to_result
        else:
            raise NotImplementedError('Defining both q and sft checkpoints are not supported!')
            
        print('Saving generated data to {}'.format(args.path_to_result))
        OUTPUT_DIR = './outputs/'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        args.path_to_result = os.path.join(OUTPUT_DIR, args.path_to_result)
        jdump(list_dict_data, args.path_to_result)