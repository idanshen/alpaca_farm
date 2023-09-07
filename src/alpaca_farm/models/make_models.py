from collections import OrderedDict

import torch
from peft.utils.save_and_load import get_peft_model_state_dict

from .. import common
import reward_model as reward_model_module

# TODO (seungwook): not very good design b/c it masks what arguments are being passed and hardcodes them
def make_generative_policy(args, accelerator, is_trainable=False):
    base_model = common.get_accelerate_model(
        model_name_or_path=args.policy_model_name_or_path,
        pretrained_lora_weights=args.policy_model_checkpoint_dir,
        four_bits=args.four_bits,
        use_lora=args.use_lora,
        flash_attn=args.flash_attn,
        is_trainable=is_trainable,
        accelerator=accelerator,)
    return base_model


def make_reward_model(args, accelerator, is_trainable=False):
    reward_model_config = reward_model_module.RewardConfig(backbone_model_name_or_path=args.reward_model_name_or_path)
    # for pretrained reward models that aren't lora-based
    if reward_model_config.backbone_model_name_or_path != 'huggyllama/llama-7b':
        base_reward_model = reward_model_module.RewardNoLoraModel(
            transformer_cache_dir=args.transformer_cache_dir,
            four_bits=False,
            flash_attn=args.flash_attn,
            is_trainable=is_trainable,
            config=reward_model_config,
            accelerator=accelerator)
    else:
        base_reward_model = reward_model_module.RewardModel(
            transformer_cache_dir=args.transformer_cache_dir,
            four_bits=args.four_bits,
            use_lora=args.use_lora,
            flash_attn=args.flash_attn,
            pretrained_lora_weights=args.reward_model_checkpoint_dir,
            is_trainable=is_trainable,
            config=reward_model_config,
            accelerator=accelerator)
    return base_reward_model


# returns a reward soup model given multiple lora checkpoints and their coefficients
def make_rewardsoup_model(args, accelerator, lora_checkpoints, coefs, is_trainable=False):
    def average_weights():
        weights_averaged = OrderedDict()
        for lora_ckpt, c in zip(lora_checkpoints, coefs):
            if c == 0.:
                continue
            if lora_ckpt is None:
                print("Skipping none lora_ckpt")
                continue
            
            args.policy_model_checkpoint_dir = lora_ckpt
            current_model = make_generative_policy(args, accelerator, is_trainable)
            current_weights = get_peft_model_state_dict(current_model, state_dict=None)
            
            for key in list(current_weights.keys()):
                if i == 0:
                    weights_averaged[key] = c * current_weights[key]
                else:
                    weights_averaged[key] += c * current_weights[key]
                del current_weights[key]
            del current_model
            torch.cuda.empty_cache()
            
        return weights_averaged
    
    averaged_weights = average_weights()
    torch.cuda.empty_cache()
    model = make_generative_policy(args, accelerator, is_trainable)
    model.load_state_dict(averaged_weights, strict=False) # b/c only loading lora
    
    return model