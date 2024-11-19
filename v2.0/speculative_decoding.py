import torch
from torch.nn import Module
from utils.logits_processors import LogitsProcessor, GreedyProcessor
from transformers.cache_utils import DynamicCache
from typing import List, Tuple
from tqdm import tqdm

def max_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Max function.
        x: input tensor.
    Returns:
        tensor norm(max(0, x)).
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum


@torch.no_grad()
def speculative_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    tokenizer = None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
) -> Tuple[List[int], float]:
    """
    Generate text sequence using the speculative decoding algorithm.
    Implementation of Speculative Decoding. (https://arxiv.org/pdf/2211.17192.pdf)
    
    Args:
        inputs (List[int]): input sequence of batch size 1.
        drafter (Module): drafter model.
        target (Module): target model.
        tokenizer: tokenizer (used for debugging).
        gamma (int): number of drafts generated by the drafter at each step.
        logits_processor (LogitsProcessor): logits processor for sampling.
        max_gen_len (int): maximum length of the generated sequence.
        eos_tokens_id (int or List[int]): end token id (could be multiple).
        pad_token_id (int): pad token id.
        use_cache (bool): whether to use cache.
        skip_sample_adjustment (bool): whether to skip the sample adjustment step when some drafts are discarded.
        first_target (bool): whether to run the target model before the speculative algorithm.
        debug (bool): debug mode.
    
    Returns:
        List[int]: generated sequence.
        float: acceptance rate (number of accepted drafts divided by the number of total drafts).
        
    Note: This generation methods only works for decoder-only models.
    Note bis: The drafter and target models should output the same logits shape.
    """
    
    drafter_cache, target_cache = None, None

    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
    
    drafts_accepted, drafts_speculated = .0, .0
    
    vocabulary_size = target.config.vocab_size    
        
    # prepare input tensor
    prompt_len = len(inputs)
    max_seq_length = target.config.max_position_embeddings if hasattr(target.config, 'max_position_embeddings') else (target.config.max_context_length if hasattr(target.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=target.device)
    
    current_position = prompt_len
    
    if first_target:
        # run the target model before the speculative algorithm. Allows to prefill the kvcache and get a first token.
        Mp = target(
            input_ids=input_ids[..., :current_position],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        p_p = logits_processor(Mp.logits[..., -1, :])
        t = logits_processor.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1
        
        if torch.isin(t, stop_tokens):
            # if debug:
            #     printing.end_token_found(0)
            return input_ids[0, prompt_len:current_position].tolist(), 0
        
        # if debug:
        #     printing.initial_step(t, tokenizer)
    while current_position < total_len:
    # while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)
        
        input_ids = input_ids.to(drafter.device)
        
        # generate gamma drafts
        for k in range(corrected_gamma):
            Mq = drafter(
                input_ids=input_ids[..., :current_position + k],
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Mq.past_key_values
            
            draft_logits = Mq.logits[..., -1, :]
            draft_probs = logits_processor(draft_logits)
            q[0, k] = draft_probs.to(target.device)
            xi = logits_processor.sample(draft_probs)
            input_ids[0, current_position + k] = xi
        drafts_speculated += corrected_gamma
        input_ids = input_ids.to(target.device)
        
        # run target model on drafts and get logits of the previous tokens plus one more token
        Mp = target(
            input_ids=input_ids[..., :current_position + corrected_gamma],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        draft_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :] # [1, corrected_gamma, vocab_size]
        p = logits_processor(draft_logits) # [1, gamma, vocab_size]
        
        # compute the last accepted draft position (rejection sampling)
        r = torch.rand(corrected_gamma, device=target.device)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                n = i
                break
        
        drafts_accepted += n
        
        # check if the end token is in the drafts
        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:current_position + n], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[0, 1].item()
            return input_ids[0, prompt_len:current_position + stop_location + 1].tolist(), drafts_accepted / drafts_speculated

        # adjust the distribution from Mp
        if n == corrected_gamma:
            p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
            p_p = logits_processor(p_p)
        else:
            if not skip_sample_adjustment:
                p_p = max_fn(p[..., n, :] - q[0, n, :])
            else:
                p_p = p[..., n, :]
        x = logits_processor.sample(p_p)
            
        input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id # removing all the uses information
        input_ids[0, current_position + n] = x
       
        current_position += n + 1
        
        if torch.isin(x, stop_tokens):
            return input_ids[0, prompt_len:current_position].tolist(), drafts_accepted / drafts_speculated
    
    return input_ids[0, prompt_len:].tolist(), drafts_accepted / drafts_speculated