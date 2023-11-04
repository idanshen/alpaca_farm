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

import os
from os.path import exists, join, isdir
import re
import time
import types
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union
from optimum.bettertransformer import BetterTransformer

import accelerate
import torch
import torch.distributed as dist
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from accelerate.utils import convert_outputs_to_fp32, is_torch_version
from torch import nn
from torch._dynamo import OptimizedModule
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from transformers.trainer import WEIGHTS_NAME#, is_deepspeed_zero3_enabled
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
import bitsandbytes as bnb
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel, PeftModelForCausalLM, LoraModel, prepare_model_for_kbit_training
)
from peft.tuners.lora import LoraLayer

from . import constants, logging, utils
from .models.reward_model import RewardModel
from .types import AnyPath, AnyPathOrNone

logger = logging.get_logger(__name__)


def apex_is_installed():
    try:
        import apex

        return True
    except ImportError as _:
        return False


def flash_attn_is_installed():
    try:
        import flash_attn

        return True
    except ImportError as _:
        return False


class staggered_object_creation(object):
    """
    Objection creation in a distributed setting could be very RAM-intensive.

    This function staggers the creation of objects on odd and even ranks, so that not all objects
    are created at once.

    Assumes local_rank == -1 means no distributed training.
    """

    def __init__(self, local_rank: int, world_size: int):
        super().__init__()
        self.local_rank = local_rank
        self.world_size = world_size

    def __enter__(self, *args, **kwargs):
        del args, kwargs
        if self.world_size > 1 and self.local_rank % 2 == 0:
            dist.barrier()
        return self

    def __exit__(self, *args, **kwargs):
        del args, kwargs
        if self.world_size > 1:
            if self.local_rank % 2 == 1:
                dist.barrier()
            dist.barrier()  # Final safety barrier.

    def __call__(self, func):
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator


def make_generative_lm(
    model_name_or_path: str,
    flash_attn: bool,
    fp16: Optional[bool] = None,
    bf16: Optional[bool] = None,
    mixed_precision: Optional[str] = None,
    **kwargs,
):
    if fp16 is None:
        fp16 = mixed_precision == "fp16"
    if bf16 is None:
        bf16 = mixed_precision == "bf16"

    if flash_attn and not fp16 and not bf16:
        logger.warning(
            "Flash attention does not support fp32. Reverting to standard attention.", main_process_only=True
        )
        flash_attn = False

    if flash_attn and flash_attn_is_installed():
        from .flash_models import flash_llama

        model_cls = flash_llama.LlamaForCausalLM
    else:
        model_cls = transformers.LlamaForCausalLM

    return model_cls.from_pretrained(model_name_or_path, **kwargs)


def get_accelerate_model(
    model_name_or_path: str,
    accelerator: accelerate.Accelerator,
    flash_attn: bool = False,
    four_bits: bool = True,
    gradient_checkpointing: bool = True,
    transformer_cache_dir: Optional[str] = None,
    use_lora: bool = True,
    lora_r: int = 60,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    pretrained_lora_weights: Optional[str] = None,
    is_trainable: bool = True,
    **kwargs,
):

    print(f'loading base model {model_name_or_path}...')

    compute_dtype = torch.float32
    if accelerator.mixed_precision == 'bf16':
        compute_dtype = torch.bfloat16
    elif accelerator.mixed_precision == 'fp16':
        compute_dtype = torch.float16
    else:
        assert accelerator.mixed_precision == 'fp32'
    
    if four_bits:
        model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=four_bits,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=four_bits,
            bnb_4bit_quant_type='nf4'  # by default, options are {'fp4', 'nf4'}
        ),
        trust_remote_code=False,  # Set True to enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained.
        cache_dir=transformer_cache_dir,
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            trust_remote_code=False,  # Set True to enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained.
            cache_dir=transformer_cache_dir,
        )

    if flash_attn:
        print("Using Flash Attention. Notice that this feature requires per device batch size 1.")
        model = BetterTransformer.transform(model)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if use_lora:
        if is_trainable and four_bits:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
        modules = find_all_linear_names(four_bits, model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        if pretrained_lora_weights is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, pretrained_lora_weights, is_trainable=is_trainable)
        else:
            print(f'adding LoRA modules...')
            model = get_peft_model(model, config)

    # compile the model for fast inference
    # model = torch.compile(model)

    # wrap with accelerator for mixed precision
    accelerator.prepare(model)

    return model

# same as get_accelerate_model but for sequence classification models that aren't lora-based
def get_accelerate_sc_model(
    model_name_or_path: str,
    accelerator: accelerate.Accelerator,
    flash_attn: bool = False,
    four_bits: bool = True,
    gradient_checkpointing: bool = True,
    transformer_cache_dir: Optional[str] = None,
    is_trainable: bool = True,
    **kwargs,
):
    print(f'loading base model {model_name_or_path}...')

    if four_bits:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            device_map='auto',
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=four_bits,
                bnb_4bit_compute_dtype=torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float16,
                bnb_4bit_use_double_quant=four_bits,
                bnb_4bit_quant_type='nf4'  # by default, options are {'fp4', 'nf4'}
            ),
            trust_remote_code=False,
            cache_dir=transformer_cache_dir,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            device_map='auto',
            trust_remote_code=False,
            # Set True to enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained.
            cache_dir=transformer_cache_dir,
        )

    if flash_attn:
        print("Using Flash Attention. Notice that this feature requires per device batch size 1.")
        model = BetterTransformer.transform(model)

    # turning gradients on/off depending on whether it is_trainable
    for p in model.parameters():
        p.requires_grad = is_trainable

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if is_trainable:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    
    # skip mixed precision if t5 b/c of nan issue
    if 't5' in model_name_or_path.lower():
        return model

    # wrap with accelerator for mixed precision
    accelerator.prepare(model)

    return model

def find_all_linear_names(four_bit, model):
    cls = bnb.nn.Linear4bit if four_bit else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def let_model_save_mem_when_zero_grad(model: nn.Module):
    def new_zero_grad(self, set_to_none: bool = True) -> None:
        r"""Sets gradients of all model parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead."
            )

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    # Make zero_grad `set_to_none=True` by default.
    # Need this runtime method patching, since self is used within zero_grad.
    model.zero_grad = types.MethodType(new_zero_grad, model)
    return model


def save_peft_model(model: PeftModel, peft_model_path: str):
    model.save_pretrained(peft_model_path)


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, model, give_rw_access=True, rank0_only=True
):
    """Collects the state dict and dump to disk."""
    now = time.perf_counter()

    if isinstance(model, PeftModel):
        peft_model_path = os.path.join(output_dir, "adapter_model")
        save_peft_model(model, peft_model_path)
    elif isinstance(model, RewardModel):
        peft_model_path = os.path.join(output_dir, "adapter_model")
        save_peft_model(model.backbone_model, peft_model_path)
        # Saving final layer
        torch.save(model.reward_head, os.path.join(peft_model_path, "reward_head.pt"))

    if trainer.fsdp is not None:
        # NOTE(rtaori): technically should be rank0_only=True (otherwise duplicates model in RAM),
        # but currently there seems to be a bug in FSDP that causes it to hang.
        # Migration to Pytorch 2 should fix this.
        # Once we migrate, we can also implement more efficient loading:
        # https://github.com/pytorch/pytorch/blob/master/torch/distributed/fsdp/api.py#L286-L295
        # NOTE(tianyi): tested on sphinx6, seems to work fine with rank0_only=False
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
        with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = trainer.model.state_dict()
            if trainer.args.should_save:
                trainer._save(output_dir, state_dict=state_dict)  # noqa

    elif trainer.deepspeed is not None:
        # --- The stuff below is almost a copy from transformers.trainer.Trainer.save_model (transformers==4.27.3) ---
        # this takes care of everything as long as we aren't under zero3
        if trainer.args.should_save:
            trainer._save(output_dir)

        # if is_deepspeed_zero3_enabled():
        #     # It's too complicated to try to override different places where the weights dump gets
        #     # saved, so since under zero3 the file is bogus, simply delete it. The user should
        #     # either use deepspeed checkpoint to resume or to recover full weights use
        #     # zero_to_fp32.py stored in the checkpoint.
        #     if trainer.args.should_save:
        #         file = os.path.join(output_dir, WEIGHTS_NAME)
        #         if os.path.isfile(file):
        #             logger.warning(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
        #             os.remove(file)

            # now save the real model if stage3_gather_16bit_weights_on_model_save=True
            # if false it will not be saved.
            # This must be called on all ranks
            if not trainer.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                logger.warning(
                    "deepspeed.save_16bit_model didn't save the model, since"
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                trainer.deepspeed.save_checkpoint(output_dir)
                # --- End of shameless copy ---

                # Auto-convert the checkpoint to fp32 for easier downstream use.
                # Only rank0 shall do the checkpoint conversion to prevent race conditions.
                if trainer.args.should_save:
                    try:
                        os.system(
                            f"python {output_dir}/zero_to_fp32.py  '{output_dir}' '{output_dir}/pytorch_model.bin'"
                        )
                    except Exception as e:
                        logger.fatal(f"Failed to convert zero3 checkpoint to fp32: {e}")

    else:  # Also support saving for non-FSDP models.
        # NOTE(lxuechen): Saving and loading T5 has weird pickle issues due to device map.
        #  Wasn't able to exactly pinpoint. But saving to and loading from CPU seems to work.
        #  In principle, trainer.save_model() should do the same thing, but breaks in practice.
        #  We drop T5 support.
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

    if trainer.args.should_save:
        if give_rw_access:
            try:
                os.system(f"chmod -R a+xwr {output_dir}")
            except Exception as e:
                logger.fatal(f"Failed to give read-write access to {output_dir}: {e}")
        logger.warning(f"Saving model took {time.perf_counter() - now:.2f} seconds.")


def flatten_dict(nested, sep=".", postprocess_fn=lambda *args: args):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):  # collections.Mapping fails in py3.10.
                rec(v, prefix + k + sep, into)
            else:
                v = postprocess_fn(v)
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def unpack_dict(d: Dict, keys: Sequence[str], return_type: type = tuple) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def merge_dict(dicts: Sequence[dict], merge_fn: Callable = lambda *args: args) -> dict:
    """Merge a sequence of dicts (with the same set of keys) into a single dict."""
    if len(dicts) == 0:
        return dict()
    return {key: merge_fn([dict_[key] for dict_ in dicts]) for key in dicts[0].keys()}


def model_name_or_path_exists(model_name_or_path: AnyPath) -> bool:
    try:
        transformers.PretrainedConfig.get_config_dict(model_name_or_path)
    except OSError:
        return os.path.exists(Path(model_name_or_path) / "trainer_state.json")
    return True


def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, OptimizedModule):
        model = model._modules['_orig_mod']
    if isinstance(model, PeftModel):
        if isinstance(model.base_model, LoraModel):
            model = model.base_model.model
        else:
            model = model.base_model
    if isinstance(model, transformers.GPT2LMHeadModel) or isinstance(model, transformers.GPT2Model):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    else:
        # Hack to deal with the fact that transformers library changed the LLaMA model name.
        llama_cls = getattr(
            transformers, "LLaMAForCausalLM" if hasattr(transformers, "LLaMAForCausalLM") else "LlamaForCausalLM"
        )
        if isinstance(model, llama_cls):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)


def prepare_inputs(data: Union[torch.Tensor, Any], device: Union[str, int, torch.device]) -> Union[torch.Tensor, Any]:
    if isinstance(data, Mapping):
        return type(data)({k: prepare_inputs(v, device) for k, v in data.items()})  # noqa
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_inputs(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)  # This can break with deepspeed.
    return data


def cast_with_native_amp(func: Callable, mixed_precision: Optional[str] = None) -> Callable:
    """Almost like how huggingface accelerate cast `model.forward`."""
    if mixed_precision not in ("fp16", "bf16"):
        logger.warning(f"Unknown mixed precision mode: {mixed_precision}, falling back to fp32.")
        return func

    if mixed_precision == "fp16" and is_torch_version(">=", "1.10"):
        output_func = torch.cuda.amp.autocast(dtype=torch.float16)(func)
    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        output_func = torch.autocast(device_type=device_type, dtype=torch.bfloat16)(func)
    output_func = convert_outputs_to_fp32(output_func)
    return output_func


def prepare_model_for_custom_fn(model: nn.Module, fn_name: str, accelerator: accelerate.Accelerator) -> nn.Module:
    """Wrap a custom function of a model with the right mixed precision context.

    This function should be run on *raw* model, i.e., before wrapped into DDP or FSDP.
    """
    if accelerator.native_amp:
        # Store original function.
        original_fn_name = f"_original_{fn_name}"
        original_fn = getattr(model, fn_name)
        setattr(model, original_fn_name, original_fn)

        # New set function.
        wrapped_fn = cast_with_native_amp(original_fn, mixed_precision=accelerator.mixed_precision)
        setattr(model, fn_name, wrapped_fn)
    return model
