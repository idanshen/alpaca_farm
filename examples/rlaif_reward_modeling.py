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

import contextlib
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Literal

import transformers
from accelerate.utils import set_seed

from alpaca_farm import common, accelerate_patch,constants, data_utils, logging, utils
from alpaca_farm.models import reward_model
from alpaca_farm.reward_modeling_trainer import SoftPreferenceTrainer, compute_soft_preference_reward_modeling_metrics

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Name of or path to the base generative LM."},
    )
    checkpoint_dir: str = field(
            default=None,
            metadata={"help": "Path to the lora weights."},
        )

@dataclass
class DataArguments:
    dataset_path: str = field(default='openai/summarize_from_feedback',
                              metadata={"help": "Name of dataset (for identifier and preprocessing purposes)."})
    train_data_filpeath: str = field(default="./rlaif_results/rlaif_gpt-3.5-turbo-instruct_openai/summarize_from_feedback_data_train.json")
    validation_data_filepath: str = field(default="./rlaif_results/rlaif_gpt-3.5-turbo-instruct_openai/summarize_from_feedback_data_validation.json")
    prompt_dict_path: str = field(
        default=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
        metadata={"help": "Path to the dictionary for the prompt to format examples."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    cache_dir: str = field(default=constants.DEFAULT_CACHE_DIR)
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    flash_attn: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    label_names: List[str] = field(
        default_factory=lambda: ["choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    initialize_model_on_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether to initialize the model on CPU. "
            "If True, models on all processes will be first initialized on CPU; this is RAM-costly but faster."
        },
    )
    end_sequence_with_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to end sequences with EOS. "
            "Ending with EOS might help the reward model realize it's time to predict."
        },
    )
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "If True, loads from last check point."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )
    transformer_cache_dir: str = field(
        default=None,
        metadata={
            "help": "Path to a directory where transformers will cache the model. "
            "If None, transformers will use the default cache directory."
        },
    )
    four_bits: bool = field(default=True, metadata={"help": "If True, uses 4-bit quantization."})
    bfloat16: bool = field(default=False, metadata={"help": "If True, uses bfloat16 quantization. If lora and four_bits are True, bfloat16 is used for the lora weights."})
    use_lora: bool = field(default=True, metadata={"help": "If True, uses LoRA."})
    lora_r: int = field(default=60, metadata={"help": "LoRA local rank parameter."})
    lora_alpha: float = field(default=16, metadata={"help": "LoRA alpha parameter."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout parameter."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "If True, uses gradient checkpointing. It will require less memory but will be slower."})
    pretrained_lora_weights: str = field(default=None, metadata={"help": "Path to pretrained LoRA weights."})


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project

    # set seed for determniistic training
    set_seed(training_args.seed)

    accelerator = accelerate_patch.MyAccelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision='bf16' if training_args.bf16 else 'fp16',
        log_with=["wandb"],
        even_batches=True,  # Make sure the batch size on each device is the same.
        split_batches=False,  # Don't break a batch into smaller chunks.
        step_scheduler_with_optimizer=False,  # Untie optimizer and scheduler step.
    )
    
    accelerator.init_trackers(
        training_args.wandb_project,
        init_kwargs={"wandb": {"name": training_args.run_name}},
        config=training_args.__dict__,
    )
    logger.warning(accelerator.state, main_process_only=False) 

    # config = transformers.PretrainedConfig.get_config_dict(model_args.model_name_or_path)
    config = reward_model.RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)
    model = reward_model.RewardModel(
        accelerator=accelerator,
        pretrained_lora_weights=training_args.pretrained_lora_weights,
        transformer_cache_dir=training_args.transformer_cache_dir,
        four_bits=training_args.four_bits,
        bfloat16=training_args.bfloat16,
        use_lora=training_args.use_lora,
        lora_r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        gradient_checkpointing=training_args.gradient_checkpointing,
        flash_attn=training_args.flash_attn,
        soft_preference=True,
        config=config,)
    common.let_model_save_mem_when_zero_grad(model)
    common.cast_with_native_amp(model.forward, accelerator.mixed_precision) # casting again b/c only base model is wrapped with accelerator

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # Ensure reward is always extracted at the last token embedding.
        use_fast=training_args.use_fast_tokenizer,
    )
    tokenizer.padding = training_args.padding
    tokenizer.pad_token_id = 0

    data_module = data_utils.make_soft_preference_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    trainer = SoftPreferenceTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_soft_preference_reward_modeling_metrics,
        **data_module,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    logger.warning("hooray! training finished successfully! now on to model saving.", main_process_only=True)

    trainer.evaluate()

    trainer.save_state()
    common.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, model=model)
    logger.warning("hooray again! model saving worked.", main_process_only=True)


if __name__ == "__main__":
    main()
