import os
import argparse

import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from best_of_n import run_decode_augmented
from alpaca_farm import data_utils, common
from alpaca_farm.utils import jload, jdump
from alpaca_farm.models import reward_model as reward_model_module
from alpaca_farm.rl.ppo_trainer import _make_left_padded_tokenizer
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator, alpaca_leaderboard_general


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

# Set up argparse to take in the model name and checkpoint dir
parser = argparse.ArgumentParser()
parser.add_argument("--reward_model_name_or_path", type=str, default="huggyllama/llama-7b"
                    , help="The name or path of the decoder to use")
parser.add_argument('--reward_checkpoint_dir', type=str, default=''
                    , help="The path to the checkpoint directory of the decoder (adapter weigthts)")
parser.add_argument('--output_filepath', type=str, default='./outputs.json',
                    help='The path to the output json file to evaluate the samples of')
parser.add_argument('--path_to_result', type=str, default='./results.json'
                    , help='The path to the output json file')
parser.add_argument('--per_device_batch_size', type=int, default=12
                    , help='The batch size to use for decoding')
parser.add_argument('--load_in_4_bits', type=bool, default=True
                    , help='Whether to load the model in 4 bits')
parser.add_argument('--exp_name', type=str, default='eval_outputs_rm'
                    , help='The name of the experiment')
args = parser.parse_args()

if 'OPENAI_API_KEY' not in os.environ:
    decoding_kwargs = dict(
        openai_api_key = "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",
        # openai_organization_ids = ["MIT"],
    )
    assert decoding_kwargs["openai_api_key"] is not None, "OPENAI_API_KEY not found you should set it in environment or above"
else:
    decoding_kwargs = {}

if os.path.isfile(args.output_filepath):
    list_dict_data = jload(args.output_filepath)
else:
    raise Exception('Output file(s) not found!')
    

print("Finish generating data, start evaluating")


reward_tokenizer: transformers.PreTrainedTokenizer = _make_left_padded_tokenizer(model_name_or_path=args.reward_model_name_or_path)
reward_model = make_reward_model(args=args)
data_module: dict = data_utils.make_eval_data_module(tokenizer=reward_tokenizer)
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
generated_data = evaluate_data(args, policy, policy_tokenizer, reward_model, reward_tokenizer, data_loader)
jdump(generated_data, args.output_file)
