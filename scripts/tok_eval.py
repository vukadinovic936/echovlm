"""
A simple script just comparing our tokenizer vs gpt tokenizer
"""
from echovlm import tokenizer
import pandas as pd
import tiktoken

# create tokenizer
echovlm_tok = tokenizer.RustBPETokenizer.from_directory("assets/tokenizer")
gpt_tok = tiktoken.encoding_for_model("gpt-4o")

# create a validation string
manifest = pd.read_csv("assets/manifest.csv")
reports = pd.read_pickle("assets/reports.pkl")
val_reports = [reports[study] for study in manifest[manifest['split']=='val']['study']]
echo_reports_as_string=""
for report in val_reports:
    echo_reports_as_string+=report

echovlm_compression_ratio = len(echo_reports_as_string)/len(echovlm_tok.encode(echo_reports_as_string))
gpt_compression_ratio = len(echo_reports_as_string)/len(gpt_tok.encode(echo_reports_as_string))
print(f"Our tokenizer's compression ratio is {echovlm_compression_ratio:.2f} and vocabulary size is {echovlm_tok.get_vocab_size()}")
print(f"GPT40 compression ratio is {gpt_compression_ratio:.2f} and vocabulary size is {gpt_tok.n_vocab}")