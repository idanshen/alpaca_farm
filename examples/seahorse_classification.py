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
import copy

from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers

from alpaca_farm import common, constants, data_utils, logging, utils, accelerate_patch
from alpaca_farm.models import reward_model
from transformers import Trainer, TrainerCallback

from datasets import load_metric
import numpy as np
metric = load_metric('accuracy')

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Name of or path to the base generative LM."},
    )

@dataclass
class DataArguments:
    dataset_path: str = field(default="./seahorse_data/")
    dataset_name: Literal["alpaca_human_preference", "alpaca_gpt4_preference", "alpaca_noisy_multi_preference"] = field(
        default="alpaca_noisy_multi_preference",
        metadata={"help": "Name of the dataset. Fetches the human or GPT-4 preference data."},
    )
    prompt_dict_path: str = field(
        default=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
        metadata={"help": "Path to the dictionary for the prompt to format examples."},
    )
    classification_label_key: str = field(default="question6")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=64)
    step_batch_size: int = field(default=128)
    evaluation_strategy: Literal["steps", "epoch"] = field(default="steps")
    eval_steps: int = field(default=40)
    log_level: str = field(default="info")
    logging_steps: int = field(default=5)
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    cache_dir: str = field(default=constants.DEFAULT_CACHE_DIR)
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    flash_attn: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    run_name: str = field(
        default=None,
        metadata={
            "help": "Name of the run. If None, will be set to the model name."
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
    fp16: bool = field(default=True, metadata={"help": "If True, uses fp16."})
    bf16: bool = field(default=False, metadata={"help": "If True, uses bf16."})
    four_bits: bool = field(default=True, metadata={"help": "If True, uses 4-bit quantization."})
    bfloat16: bool = field(default=False, metadata={"help": "If True, uses bfloat16 quantization. If lora and four_bits are True, bfloat16 is used for the lora weights."})
    use_lora: bool = field(default=True, metadata={"help": "If True, uses LoRA."})
    lora_r: int = field(default=60, metadata={"help": "LoRA local rank parameter."})
    lora_alpha: float = field(default=16, metadata={"help": "LoRA alpha parameter."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout parameter."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "If True, uses gradient checkpointing. It will require less memory but will be slower."})
    pretrained_lora_weights: str = field(default=None, metadata={"help": "Path to pretrained LoRA weights."})
    warmup_steps: int = field(default=20, metadata={"help": "Number of warmup steps."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay."})
    logging_first_step: bool = field(default=True, metadata={"help": "If True, logs the first step."})

    def __post_init__(self):
        self.gradient_accumulation_steps = self.step_batch_size // self.per_device_train_batch_size
        super().__post_init__()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metrics = metric.compute(predictions=predictions, references=labels)
    if np.all(predictions==0) or np.all(predictions==1):
        metrics['pearson'] = 0
    else:
        metrics['pearson'] = np.corrcoef(labels.squeeze(), predictions)[0,1]
    return metrics


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        # if control.should_evaluate:
        control_copy = copy.deepcopy(control)
        self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
        return control_copy

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    
    # preprocess classification label arg st if there is a comma, split it into a list
    if ',' in data_args.classification_label_key:
        data_args.classification_label_key = data_args.classification_label_key.split(',')

    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", model_max_length=1024)
    # model = AutoModelForSequenceClassification.from_pretrained("google/flan-t5-large")

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
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
    config = reward_model.RewardConfig(backbone_model_name_or_path='huggyllama/llama-7b')
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
        config=config,)
    common.let_model_save_mem_when_zero_grad(model)
    common.cast_with_native_amp(model.forward, accelerator.mixed_precision)
    
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')

    data_module = data_utils.make_classification_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        compute_metrics=compute_metrics,
    )
    # callback to evaluate on training dataset as well (slow)
    # trainer.add_callback(CustomCallback(trainer))

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    logger.warning("hooray! training finished successfully! now on to model saving.", main_process_only=True)

    trainer.evaluate()

    trainer.save_state()
    common.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, model=model)
    logger.warning("hooray again! model saving worked.", main_process_only=True)


if __name__ == "__main__":
    main()
