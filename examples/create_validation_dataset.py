import os

from dataclasses import dataclass, field
from typing import List, Dict, Any

import transformers
from tqdm import tqdm
import numpy as np
from alpaca_farm.utils import jload, jdump

from alpaca_farm import accelerate_patch, common, constants, data_preprocessor, data_utils
from alpaca_farm.models import reward_model as reward_model_module
from alpaca_farm.types import AnyPath, AnyPathOrNone
from torch.utils.data import DataLoader

@dataclass
class Arguments:
    policy_model_name_or_path: str = field(
        default="decapoda-research/llama-7b-hf", metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."}
    ),
    policy_model_checkpoint_dir: str = field(
        default="/data/pulkitag/models/idanshen/alpaca_farm/spf/test_3/adapter_model/", metadata={"help": "Path to a checkpoint directory."}
    ),
    reward_model_name_or_path: str = field(
        default="decapoda-research/llama-7b-hf", metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."}
    ),
    reward_model_checkpoint_dir: str = field(
        default=None, metadata={"help": "Path to a checkpoint directory."}
    ),
    four_bits: bool = field(default=True, metadata={"help": "If True, uses 4-bit quantization."})
    bfloat16: bool = field(default=False, metadata={
        "help": "If True, uses bfloat16 quantization. If lora and four_bits are True, bfloat16 is used for the lora weights."})
    use_lora: bool = field(default=True, metadata={"help": "If True, uses LoRA."})
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default="alpaca_instructions")
    train_splits: List[str] = field(default_factory=lambda: ["train"])
    eval_splits: List[str] = field(default_factory=lambda: ["validation"])
    prompt_dict_path: str = field(
        default=None,
        metadata={"help": "Path to the dictionary for the prompt to format examples."},
    )
    flash_attn: bool = field(default=False, metadata={"help": "If True, uses Flash Attention."})
    transformer_cache_dir: str = field(
        default=None,
        metadata={
            "help": "Path to a directory where transformers will cache the model. "
            "If None, transformers will use the default cache directory."
        },
    )
    query_len: int = field(default=192)
    static_dataset: bool = field(default=False, metadata={"help": "If True, uses static dataset the contains respones."})
    per_device_batch_size: int = field(default=1, metadata={"help": "Batch size per device."})
    temperature: float = field(default=0.7)
    num_completions: int = field(default=10)
    output_file: str = field(default="/data/pulkitag/models/idanshen/alpaca_farm/sft/test_5/validation_dataset.json")


def make_generative_policy(args, is_trainable=False):
    base_model = common.get_accelerate_model(
        model_name_or_path=args.policy_model_name_or_path,
        pretrained_lora_weights=args.policy_model_checkpoint_dir,
        four_bits=args.four_bits,
        bfloat16=args.bfloat16,
        use_lora=args.use_lora,
        flash_attn=args.flash_attn,
        is_trainable=is_trainable,)
    return base_model


def make_reward_model(args, is_trainable=False):
    reward_model_config = reward_model_module.RewardConfig(backbone_model_name_or_path=args.reward_model_name_or_path)
    # for pretrained reward models that aren't lora-based
    if reward_model_config.backbone_model_name_or_path != 'huggyllama/llama-7b':
        base_reward_model = reward_model_module.RewardNoLoraModel(
            transformer_cache_dir=args.transformer_cache_dir,
            four_bits=args.four_bits,
            bfloat16=args.bfloat16,
            flash_attn=args.flash_attn,
            is_trainable=is_trainable,
            config=reward_model_config, )
    else:
        base_reward_model = reward_model_module.RewardModel(
            transformer_cache_dir=args.transformer_cache_dir,
            four_bits=args.four_bits,
            bfloat16=args.bfloat16,
            use_lora=args.use_lora,
            flash_attn=args.flash_attn,
            pretrained_lora_weights=args.reward_model_checkpoint_dir,
            is_trainable=is_trainable,
            config=reward_model_config, )
    return base_reward_model


def make_left_padded_tokenizer(
    model_name_or_path: AnyPath,
    cache_dir: AnyPathOrNone = constants.DEFAULT_CACHE_DIR,
    **kwargs,
) -> transformers.PreTrainedTokenizer:
    print(f"Loading tokenizer from {model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        padding_side="left",
        **kwargs,
    )
    tokenizer.padding = "longest"
    if model_name_or_path == "huggyllama/llama-7b":
        tokenizer.pad_token_id = 0
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def generate_data(args, policy, policy_tokenizer, reward_model, reward_tokenizer, data_loader) -> List[Dict[str, Any]]:
    """Generates data using the policy and reward models. For each example in the data loader, the policy generates
     a partial response in a random length (between 1 to args.max_partial_response_length). Then, the policy generates
     args.num_completions completions to this partial response, and the reward model scores the response.
     The partial response and the average of the reward scores are saved in a json file."""

    generated_data = []

    for batch in tqdm(iter(data_loader), desc="steps",):
        queries, query_attn_masks = common.unpack_dict(
            common.prepare_inputs(batch, device=0),
            keys=("queries", "query_attn_masks"),
        )
        n = np.random.randint(1, 100)
        short_response = policy.generate(
            inputs=queries,
            attention_mask=query_attn_masks,
            do_sample=True,
            max_new_tokens=n,
            pad_token_id=policy_tokenizer.pad_token_id,
            top_p=1.0,
            top_k=0,
            temperature=args.temperature,
            num_return_sequences=1,
        )
        text_short_response = policy_tokenizer.batch_decode(short_response, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        encoded_short_response = policy_tokenizer(text_short_response, return_tensors="pt", padding=True, truncation=True)

        full_responses = policy.generate(
            inputs=encoded_short_response['input_ids'].repeat(args.num_completions,1).to(0),
            attention_mask=encoded_short_response['attention_mask'].repeat(args.num_completions,1).to(0),
            do_sample=True,
            max_new_tokens=300-n,
            pad_token_id=policy_tokenizer.pad_token_id,
            top_p=1.0,
            top_k=0,
            temperature=args.temperature,
            num_return_sequences=1,
        )
        text_full_responses = policy_tokenizer.batch_decode(full_responses, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        encoded_full_responses = reward_tokenizer(text_full_responses, return_tensors="pt", padding=True, truncation=True)
        encoded_full_responses, = common.prepare_inputs((encoded_full_responses, ), device=0)

        reward_outputs = reward_model(**encoded_full_responses)
        rewards = reward_outputs['rewards'].cpu().detach().numpy()
        generated_data.append({"text": text_short_response[0], "reward": np.mean(rewards)})

        return generated_data


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()
    policy_tokenizer: transformers.PreTrainedTokenizer = make_left_padded_tokenizer(model_name_or_path=args.policy_model_name_or_path)
    reward_tokenizer: transformers.PreTrainedTokenizer = make_left_padded_tokenizer(model_name_or_path=args.reward_model_name_or_path)
    policy = make_generative_policy(args=args)
    reward_model = make_reward_model(args=args)
    data_module: dict = data_utils.make_rl_data_module(
        tokenizer=[policy_tokenizer, reward_tokenizer], data_args=args, training_args=args
    )
    test_dataset = data_module["eval_dataset"]
    data_collator = data_module["data_collator"]
    print(f"Test dataset size: {len(test_dataset)}")
    data_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        drop_last=True,
    )
    generated_data = generate_data(args, policy, policy_tokenizer, reward_model, reward_tokenizer, data_loader)
    jdump(generated_data, args.output_file)
