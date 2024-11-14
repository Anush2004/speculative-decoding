from math import inf
import torch
from torch.nn import Module
from utils.logits_processors import LogitsProcessor, GreedyProcessor
from typing import List
from tqdm import tqdm

@torch.no_grad()
def autoregressive_generate(
    inputs: List[int],
    model: Module,
    max_gen_len: int = 40,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
) -> List[int]:
    """
    Generate text sequence autoregressively based on the input sequence.

    Args:
        inputs (List[int]): input sequence of batch size 1.
        model (Module): model to use for inference.
        max_gen_len (int): maximum length of the generated sequence.
        logits_processor (LogitsProcessor): logits processor for sampling.
        eos_token_id (int): end token id.
        pad_token_id (int): pad token id.
        use_cache (bool): whether to use cache.

    Returns:
        List[int]: generated sequence.

    Note:
        This generation methods only works for decoder-only models.
    """
    cache = None
    prompt_len = len(inputs)
    # prepare input tensor
    max_seq_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else (model.config.max_context_length if hasattr(model.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=model.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=model.device)

    list_tokens_id = (
        eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    )
    print(model.device)
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=model.device)

    for curr in tqdm(range(prompt_len, total_len)):
        o = model(input_ids[..., :curr], past_key_values=cache, use_cache=use_cache)
        logits = o.logits[..., -1, :]  # [1, vocab_size]
        probs = logits_processor(logits)  # [1, vocab_size]
        x = logits_processor.sample(probs)  # [1, 1]
        input_ids[0, curr] = x
        cache = o.past_key_values
    # print(input_ids[0])
    return input_ids[0, prompt_len : curr + 1].tolist()

