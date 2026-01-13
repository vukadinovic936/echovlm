"""
Training a tokenizer for echocardioraphy reports
the experiments showed that the best vocabulary size is 4000
"""
import pandas as pd
import time
from echovlm.tokenizer import RustBPETokenizer
import os
import torch
import re

# load reports and shuffle them
vocab_size=2000
manifest = pd.read_csv("assets/manifest.csv")
reports = pd.read_pickle("assets/reports.pkl")
train_reports = [reports[study] for study in manifest[manifest['split']=='train']['study']]
val_reports = [reports[study] for study in manifest[manifest['split']=='val']['study']]
test_reports = [reports[study] for study in manifest[manifest['split']=='test']['study']]

def text_iterator():
    for report in train_reports:
        yield report

text_iter = text_iterator()
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")
# -----------------------------------------------------------------------------
# Save the tokenizer to disk
tokenizer.save("assets/tokenizer")
# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
# save token bytes
token_bytes_path = "assets/token_bytes.pt"
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)

token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
print(
    f"[Tokenizer training] time={train_time}, "
    f"special_tokens={len(special_set)}, "
    f"bytes(min/mean/max/std)="
    f"{int(token_bytes_nonzero.min().item())}/"
    f"{token_bytes_nonzero.mean().item():.3f}/"
    f"{int(token_bytes_nonzero.max().item())}/"
    f"{token_bytes_nonzero.std().item():.3f}"
)
echo_reports_as_string=""
for report in val_reports:
    echo_reports_as_string+=report
## also now evaluate to see what's the compression ratio
print(f"Compression ratio is {len(echo_reports_as_string)/len(tokenizer.encode(echo_reports_as_string))}")

# -----------------------------------------------------------------------------
# Calculate tokens per report to get a good sense of the distribution
tokens_per_report = torch.tensor(
    [len(tokenizer.encode(report)) for report in val_reports],
    dtype=torch.float32,
)

print(
    "tokens per report (min/mean/max/std)="
    f"{int(tokens_per_report.min().item())}/"
    f"{tokens_per_report.mean().item():.2f}/"
    f"{int(tokens_per_report.max().item())}/"
    f"{tokens_per_report.std().item():.2f}"
)