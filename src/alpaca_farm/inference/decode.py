# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import dataclasses
import math
import sys
from typing import Callable, List, Optional, Sequence, Tuple, Union
import os
from argparse import Namespace

import einops
import torch
import tqdm
import transformers
from transformers import LogitsProcessorList
from peft import PeftModel

from .. import common, constants, distributed_utils, logging, torch_ops, utils
from ..models.rl_models import make_qfunction_with_base_model, AutoregressiveQfunction

logger = logging.get_logger(__name__)

class QLogitsProcessor(transformers.LogitsProcessor, torch.nn.Module):
    """
    A hack class to process logits to integrate learned Q values

    (currently assumes that the tokenizer for the policy and q model are the same)
    """
    def __init__(self, q_model: AutoregressiveQfunction, beta: float):
        # call super init
        super().__init__()
        self.q_model = q_model # assumes that q model is already moved to device (whether on 1 device or multiple)
        self.beta = beta
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # TODO (seungwook): may need to pass in mask as well?
        q_outputs = self.q_model(input_ids, only_last=True)
        return scores + self.beta * q_outputs['qvalues'].squeeze()
    
    def _apply(self, fn):
        super()._apply(fn)
        self.q_model = self.q_model._apply(fn)
        return self


@dataclasses.dataclass
class NullCharCleanUp(object):
    def __call__(self, string: str):
        return string.replace("\x00", "")

    def __repr__(self):
        return "NullCharCleanUp cleans up the NULL chars to prevent db write failures due to encoding discrepancy."


def load_model_and_tokenizer_for_inference(
    model_name_or_path: str,
    cache_dir=constants.DEFAULT_CACHE_DIR,
    model_cls=transformers.AutoModelForCausalLM,
    model_kwargs: Optional[dict] = None,
    tokenizer_kwargs: Optional[dict] = None,
    resize_token_embeddings_if_mismatch=True,
    load_in_4_bits=False,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Load huggingface model and tokenizer from path or with name for inference.

    This function should only be used for decoding or reward scoring.

    Notes:
        - This function is only guaranteed to work correctly when loading admissible model families,
            i.e., opt and llama.
        - Loaded models are in eval mode.
        - By default, this function internally shrinks the model embedding size to avoid generating out of vocab tokens.
            Models like OPT are by default created with embedding size that's divisible by 64, even though the vocab
            size is not. This is to help with training speed, but can be problematic when generating, i.e., there is
            a low probability of generating out of vocab ids (especially for untrained models).
        - By default, loaded models are on the device specified by LOCAL_RANK or cpu.
            - This behavior can be overridden by passing `device_map` to model_kwargs.
        - By default, loaded tokenizers are slow tokenizers in left padding mode.
            - This behavior can be overridden by passing `use_fast` and `padding_side` to tokenizer_kwargs.
    """
    logger.warning(f"Loading model for inference: {model_name_or_path}")

    local_rank, world_size = distributed_utils.setup()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    default_model_kwargs = dict(low_cpu_mem_usage=True, device_map={"": device}, cache_dir=cache_dir)
    if model_kwargs is None:
        model_kwargs = default_model_kwargs
    else:
        default_model_kwargs.update(model_kwargs)  # Make possible overriding default_model_kwargs.
        model_kwargs = default_model_kwargs

    default_tokenizer_kwargs = dict(padding_side="left", use_fast=False, cache_dir=cache_dir)
    if tokenizer_kwargs is None:
        tokenizer_kwargs = default_tokenizer_kwargs
    else:
        default_tokenizer_kwargs.update(tokenizer_kwargs)
        tokenizer_kwargs = default_tokenizer_kwargs

    if load_in_4_bits:
        assert checkpoint_dir is not None, "checkpoint_dir (path to lora weights) must be specified when load_in_4_bits is True"
        model = common.get_accelerate_model(model_name_or_path, pretrained_lora_weights=checkpoint_dir, flash_attn=False, is_trainable=False, **model_kwargs).eval()
        common.let_model_save_mem_when_zero_grad(model)
    else:
        model = model_cls.from_pretrained(model_name_or_path, **model_kwargs).eval()

    # TODO (seungwook): this may need to be fixed depending on the model b/c currently assumes llama for pad_token_id=0
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    tokenizer.padding = 'longest'
    tokenizer.pad_token_id = 0

    return model, tokenizer


@dataclasses.dataclass
class HFDecodingArguments:
    """Only the core args for decoding with HF models."""

    top_p: float = 0.9
    top_k: int = 0
    temperature: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    max_new_tokens: int = 100  # This is aligned with `openai_utils.OpenAIDecodingArguments`.
    num_return_sequences: int = 1


@torch.inference_mode()
def decode_prompts_with_huggingface_given_model(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompts: Sequence[str],
    decoding_args: HFDecodingArguments,
    per_device_batch_size=20,
    mixed_precision: Optional[str] = None,
    max_instances=sys.maxsize,
    pad_to_length=2048,  # Force pad to this length for distributed communication to work.
    tf32=True,
    force_multisample_format: bool = False,
    cleanup_funcs: Optional[Sequence[Callable]] = (NullCharCleanUp(),),
    divide_work: bool = True,
    internal_batch_return_sequences: Optional[int] = None,
    seed: Optional[int] = None,
    communication_num_chunks=1,
    tokenization_batch_size=1000,
    logits_processor: Optional[transformers.LogitsProcessor] = None,
    **decoding_kwargs,
) -> Union[List[List[str]], List[str]]:
    """Decode from a given model a sequence of string prompts."""
    if seed is not None:
        utils.manual_seed(seed)

    model.eval() # TODO (seungwook): is there a reason why this is missing all over decode?
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32  # noqa

    local_rank, world_size = distributed_utils.setup()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    model.generate = common.cast_with_native_amp(model.generate, mixed_precision=mixed_precision)
    logger.warning(f"mixed_precision = {mixed_precision}")

    generate_kwargs = copy.deepcopy(decoding_args.__dict__)
    generate_kwargs.update(
        dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, synced_gpus=world_size > 1)
    )
    generate_kwargs.update(decoding_kwargs)  # Possibly overwrite default values for `pad_token_id` and `eos_token_id`.

    prompts = prompts[:max_instances]
    ori_data_size = len(prompts)

    # Make the prompts set a multiple of world_size * per_device_batch_size by padding with the last prompt.
    if world_size > 1 and divide_work:
        multiple_of = world_size * per_device_batch_size
    else:
        multiple_of = per_device_batch_size
    new_data_size = multiple_of * int(math.ceil(ori_data_size / multiple_of))
    new_prompts = list(prompts) + [prompts[-1]] * (new_data_size - ori_data_size)

    if world_size > 1 and divide_work:  # divide into chunks
        per_worker_size = new_data_size // world_size
        new_prompts = new_prompts[local_rank * per_worker_size : (local_rank + 1) * per_worker_size]
    # TODO(lxuechen): Refactor to tokenize upfront. This way we can pad with tokenizer, and not worry ourselves.

    completions = []
    for batch_idx, start_idx in tqdm.tqdm(
        enumerate(range(0, len(new_prompts), per_device_batch_size)),  # Increase the index by the actual batch size.
        desc="decoding batches",
        total=len(new_prompts) // per_device_batch_size,
        disable=not distributed_utils.is_main_process(),
    ):
        batch = new_prompts[start_idx : start_idx + per_device_batch_size]

        source = tokenizer(batch, return_tensors="pt", padding=True)
        source = common.prepare_inputs(source, device=device)
        inputs, attention_mask = source.input_ids, source.attention_mask

        # if batch_idx == 0:  # FSDP is buggy; we do a forward pass first to make it happy
        #     model(input_ids=inputs, attention_mask=attention_mask)

        if (
            internal_batch_return_sequences is not None
            and internal_batch_return_sequences < decoding_args.num_return_sequences
        ):
            # we batch along the num_return_sequences dimension to avoid OOM errors
            # usually, return_sequences is dimension (NxR, L) where N is the batch size and R is the number of
            # return sequences
            # we split this into batches of size (NxR', L) where R' is the number of return sequences in each batch
            batch_generate_kwargs = copy.deepcopy(generate_kwargs)
            # initialize the list of return sequences for each prompt
            sequences = []
            for internal_start_idx in range(
                0, generate_kwargs["num_return_sequences"], internal_batch_return_sequences
            ):
                internal_batch_size = batch_generate_kwargs["num_return_sequences"] = min(
                    internal_batch_return_sequences, generate_kwargs["num_return_sequences"] - internal_start_idx
                )
                internal_batch_sequences = model.generate(
                    inputs=inputs,
                    attention_mask=attention_mask,
                    logits_processor=LogitsProcessorList([logits_processor]) if logits_processor else None,
                    **batch_generate_kwargs,
                )
                if not model.config.is_encoder_decoder:
                    internal_batch_sequences = internal_batch_sequences[:, inputs.shape[1] :]
                internal_batch_sequences = torch_ops.right_pad(
                    internal_batch_sequences,
                    (internal_batch_sequences.size(0), pad_to_length),
                    value=tokenizer.pad_token_id,
                )
                # einops rearange (n d) l -> n d l
                internal_batch_sequences = einops.rearrange(
                    internal_batch_sequences, "(n d) l -> n d l", d=internal_batch_size
                )
                # append the return sequences for each prompt
                sequences.append(internal_batch_sequences)
            # concatenate the return sequences for each prompt
            sequences = torch.cat(sequences, dim=1)
            sequences = einops.rearrange(
                sequences,
                "n d l -> (n d) l",
            )
        else:
            if internal_batch_return_sequences is not None:
                logger.warning(
                    f"internal_batch_return_sequences ({internal_batch_return_sequences}) >= "
                    f"num_return_sequences ({decoding_args.num_return_sequences}). Not batching over return sequences."
                )

            sequences = model.generate(inputs=inputs, attention_mask=attention_mask, logits_processor=LogitsProcessorList([logits_processor]) if logits_processor else None, **generate_kwargs)
            if not model.config.is_encoder_decoder:
                sequences = sequences[:, inputs.shape[1] :]
            sequences = torch_ops.right_pad(sequences, (sequences.size(0), pad_to_length), value=tokenizer.pad_token_id)

        out_of_bound_mask = sequences >= len(tokenizer)
        if out_of_bound_mask.any():
            logger.fatal(f"Found tokens outside the vocabulary: {sequences[out_of_bound_mask]}")
        completions.append(sequences.cpu())

    completions = torch.cat(completions, dim=0)
    if world_size > 1 and divide_work:
        torch.cuda.empty_cache()
        logger.info(f"RANK {local_rank} starting all_gather with {communication_num_chunks} communication_num_chunks")
        mine = einops.rearrange(completions, "(n d) l -> n d l", d=generate_kwargs["num_return_sequences"])
        chunks = torch.chunk(mine, chunks=communication_num_chunks, dim=1)
        all_chunk_list = [
            distributed_utils.all_gather_and_cat(chunk.contiguous().to(device), dim=0).cpu() for chunk in chunks
        ]
        completions = torch.cat(all_chunk_list, dim=1)
        completions = einops.rearrange(completions, "n d l -> (n d) l")

    logger.info(
        f"RANK {local_rank} Start tokenizer batch decoding {completions.size(0)} sequences", main_process_only=False
    )
    # chunk completions into chunks of 1000 and tokenize
    text_sequences = []
    for start_idx in tqdm.trange(0, completions.size(0), tokenization_batch_size):
        text_sequences.extend(
            tokenizer.batch_decode(
                completions[start_idx : start_idx + tokenization_batch_size],
                skip_special_tokens=True,
            )
        )

    if cleanup_funcs is not None:
        for cleanup_func in cleanup_funcs:
            text_sequences = [cleanup_func(s) for s in text_sequences]

    logger.info(f"RANK {local_rank} Finished tokenizer batch decoding and cleaning", main_process_only=False)
    # convert the list into a nested list of consecutive `num_return_sequences` items, if > 1.
    if decoding_args.num_return_sequences > 1 or force_multisample_format:
        text_sequences = [
            text_sequences[i : i + decoding_args.num_return_sequences]
            for i in range(0, len(text_sequences), decoding_args.num_return_sequences)
        ]

    text_sequences = text_sequences[:ori_data_size]

    return text_sequences


def decode_prompts_with_huggingface(
    model_name_or_path: str,
    prompts: Sequence[str],
    decoding_args: HFDecodingArguments,
    cache_dir=constants.DEFAULT_CACHE_DIR,
    per_device_batch_size=20,
    mixed_precision: Optional[str] = None,
    max_instances=sys.maxsize,
    pad_to_length=2048,  # Force pad to this length for distributed communication to work.
    tf32=True,
    load_in_4_bits: bool = False,
    checkpoint_dir: Optional[str] = None,
    q_checkpoint_dir: Optional[str] = None,
    flash_attn: bool = False,
    model_and_tokenizer: Optional[Tuple] = None,
    force_multisample_format: bool = False,
    seed: Optional[int] = None,
    communication_num_chunks: int = 1,
    beta: float = 1.0,
    **decoding_kwargs,
) -> Union[List[List[str]], List[str]]:
    """Decode from a huggingface model given a sequence of string prompts.

    Args:
        prompts: A sequence of string prompts.
        decoding_args: Decoding arguments.
        model_name_or_path: The name or path of the huggingface model. If it is a path, the directory location should also store
            the tokenizer.
        per_device_batch_size: The batch size per device for decoding.
        cache_dir: The directory to cache the huggingface model.
        checkpoint_dir: The directory to load the policy checkpoint adapter from.
        q_checkpoint_dir: The directory to load the q checkpoint adapter from.
        beta: Beta value for q value estimaator weighting.
        mixed_precision: Whether to use mixed precision. If None, no casting will be performed.
        max_instances: The maximum number of prompts to decode.
        pad_to_length: The token length to pad the prompts. This is necessary for and only used in distributed decoding.
        tf32: Whether to use tensorfloat32 for matrix multiplication.
        flash_attn: Whether to use flash attention or not
        force_multisample_format: Whether to force the outputs to be in the multisample format.
        seed: The random seed. If None, this function is generally not deterministic, unless the seed is fixed outside.
        communication_num_chunks: Number of chunks to create for final communication.
            Increase this to reduce the size of the chunk per communication.
        **decoding_kwargs: Misc keyword args for `model.generate`.
            Setting values here may override the values given by `decoding_args`.

    Returns:
        A list of string responses, if `num_return_sequences` is 1 and not `force_multisample_format`;
        otherwise, a list of lists of string responses.
    """
    if model_and_tokenizer is not None:
        model, tokenizer = model_and_tokenizer
    else:
        model, tokenizer = load_model_and_tokenizer_for_inference(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            model_kwargs=dict(torch_dtype=utils.convert_str_dtype_to_torch_dtype(mixed_precision)),
            load_in_4_bits=load_in_4_bits,
            checkpoint_dir=checkpoint_dir,
        )
        
        if flash_attn:
            print("Using Flash Attention. Notice that this feature requires per device batch size 1.")
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
    
    # TODO (seungwook): assumes that the policy and q model base are the same (may need to change)
    qlogits_processor = None
    if q_checkpoint_dir is not None:
        q_model, q_tokenizer = load_model_and_tokenizer_for_inference(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            model_kwargs=dict(torch_dtype=utils.convert_str_dtype_to_torch_dtype(mixed_precision)),
            load_in_4_bits=load_in_4_bits,
            checkpoint_dir=q_checkpoint_dir,
        )
        decoding_kwargs = Namespace(**decoding_kwargs)
        q_model = make_qfunction_with_base_model(decoding_kwargs, q_model, q_tokenizer)
        # q_model.load_state_dict(torch.load(os.path.join(q_checkpoint_dir, 'adapter_model/q_head.pt'), map_location=q_model.device))
        # TODO (seungwook): depending on the type of q head (weights or whole pickled model), load differently
        q_model.load_q_head(os.path.join(q_checkpoint_dir, 'adapter_model/q_head.pt'))

        if flash_attn:
            q_model = BetterTransformer.transform(q_model)
    
        qlogits_processor = QLogitsProcessor(q_model=q_model, beta=beta)

    return decode_prompts_with_huggingface_given_model(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        decoding_args=decoding_args,
        per_device_batch_size=per_device_batch_size,
        mixed_precision=mixed_precision,
        max_instances=max_instances,
        pad_to_length=pad_to_length,
        tf32=tf32,
        force_multisample_format=force_multisample_format,
        seed=seed,
        communication_num_chunks=communication_num_chunks,
        logits_processor=qlogits_processor,
        **decoding_kwargs,
    )
