from .. import common
import reward_model as reward_model_module

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