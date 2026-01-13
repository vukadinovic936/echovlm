"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes: The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""
import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from echovlm.common import compute_init

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula)
    except Exception as e:
        signal.alarm(0)
        return None
# -----------------------------------------------------------------------------
class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0 #current position in time in ther cache

    def reset(self):
        self.pos=0
    
    def get_pos(self):
        return self.pos
    
    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate 
        multiple samples in parallel from there
        """
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                # num_layers, batch_size, num_heads, head_dim must match
                assert dim1 == dim2, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size can be expanded
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len: self must be longer than other
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        # 2) Initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) update the pos
        self.pos = other.pos
    
    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos+T_add
        # Dynamically grow the cahce if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # as much as we need plus buffer of 1024
            t_needed = (t_needed + 1023) & ~1023 # then round up to the nearest multiple of 1024
            current_shape = list(self.kv_cache.shape)
            current_shape[4] = t_needed
            self.kv_cache.resize_(current_shape)
        # Insert k,v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # Return the full cached keys/values up to the current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # Increment pos after the last layer of the Transformer process
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B,1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequqnece for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, study_embeddings=None, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache"""
        assert isinstance(tokens, list) and isinstance(tokens[0],int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool we use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        eos = get_special("<|eos|>")
        
        if study_embeddings is not None:
            # --- prepare study embeddings ---
            study_embeddings = study_embeddings.to(device)
            study_embeddings = torch.nn.functional.normalize(study_embeddings, dim=-1)
            # (N_videos, D) -> (1, N_videos, D)
            study_embeddings = study_embeddings.unsqueeze(0)

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads":m.n_kv_head, "head_dim":m.n_embd // m.n_head, "num_layers":m.n_layer}
        seq_len_prefill = len(tokens)
        if study_embeddings is not None:
            seq_len_prefill+=study_embeddings.shape[1] # N_videos + T
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=seq_len_prefill,
            **kv_model_kwargs
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids,
                                    study_embeddings=study_embeddings,
                                    kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k) # (B,1)
        sampled_tokens = next_ids[:,0].tolist()

        # 2) Replicate the KV cache for each sample/row (we might want to do multiple generations/variations of the output with the same prompt)
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        if study_embeddings is not None:
            kv_length_hint+=study_embeddings.shape[1]
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around


        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # Stp condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break
                
            # Get sampled tokens  - either from prefil or forward pass
            if first_iteration:
                #Use the tokens we already sampled from prefill
                sampled_tokens = [sampled_tokens[0]] * num_samples # Broadcast first token to all rows
                # TODO: we should sample a token for each row instead of broadcasting
                first_iteration = False
            else: 
                # Forward the model and get the next token for each row
                logits = self.model.forward(ids, study_embeddings=None, kv_cache=kv_cache_decode) # (B,T,vocab_size)
                logits = logits[:, -1, :] #(B, vocab_size) at last time step
                next_ids = sample_next_token(logits, rng, temperature, top_k)
                sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update satte, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)

                # On <|assistant_end|> or <|eos|>, mark the row as completed
                if next_token == assistant_end or next_token == eos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1
            # Prepare ids for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)               

    def generate_batch(self, tokens, study_embeddings=None, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        eos = self.tokenizer.encode_special("<|eos|>")
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0]*len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, study_embeddings, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == eos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


