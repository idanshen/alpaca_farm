import transformers

from .. import constants
from ..types import AnyPath, AnyPathOrNone

MODELS_MAX_LENGTH = {
    # 'huggyllama/llama-7b': 2048,
    'meta-llama/Llama-2-7b-hf': 4096,
    'gpt2': 1024,
}

def _make_padded_tokenizer(
    model_name_or_path: AnyPath,
    cache_dir: AnyPathOrNone = constants.DEFAULT_CACHE_DIR,
    **kwargs,
) -> transformers.PreTrainedTokenizer:
    print(f"Loading tokenizer from {model_name_or_path}")
    
    # if using flan t5 classification model, use default tokenizer
    if 'flant5' in model_name_or_path:
        model_name_or_path = 'google/flan-t5-large'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=MODELS_MAX_LENGTH.get(model_name_or_path, 1024),
        cache_dir=cache_dir,
        **kwargs,
    )

    if 'padding_side' in kwargs:
        tokenizer.padding_side = 'left' if kwargs['padding_side'] == 'left' else 'right'

    if 'padding' in kwargs:
        tokenizer.padding = kwargs['padding']
    else:
        tokenizer.padding = "longest"
        
    if 'llama' in model_name_or_path:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
    return tokenizer
