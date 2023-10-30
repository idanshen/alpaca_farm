import transformers

from .. import constants
from ..types import AnyPath, AnyPathOrNone


def _make_padded_tokenizer(
    model_name_or_path: AnyPath,
    cache_dir: AnyPathOrNone = constants.DEFAULT_CACHE_DIR,
    **kwargs,
) -> transformers.PreTrainedTokenizer:
    print(f"Loading tokenizer from {model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        **kwargs,
    )

    if 'padding_side' in kwargs:
        tokenizer.padding_side = 'left' if kwargs['padding_side'] == 'left' else 'right'

    if 'padding' in kwargs:
        tokenizer.padding = kwargs['padding']
    else:
        tokenizer.padding = "longest"
        
    if model_name_or_path == "huggyllama/llama-7b":
        tokenizer.pad_token_id = 0
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
    return tokenizer