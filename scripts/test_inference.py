from torch.utils.data import DataLoader
from echollm import dataset
from echollm import tokenizer
import pandas as pd
import torch
from echollm.gpt import GPTConfig,GPT
from echollm.common import compute_init,autodetect_device_type,print0,compute_cleanup,get_latest_checkpoint
from echollm.loss_eval import evaluate_bpb
from contextlib import nullcontext
import os
import time
from echollm.engine import Engine
import numpy as np
import glob

checkpoint_path = get_latest_checkpoint("assets")
checkpoint = torch.load(checkpoint_path)
device = torch.device("cuda")
model_config = GPTConfig(**checkpoint['model_config'])
model = GPT(model_config)
model.load_state_dict(checkpoint['model'])
model.to(device)
tokenizer = tokenizer.RustBPETokenizer.from_directory("assets/tokenizer")
engine = Engine(model, tokenizer)

prompt=""
tokens = tokenizer(prompt, prepend="<|bos|>")
sample_embeddings = torch.load("assets/embeddings.pt")[0].unsqueeze(0)
sample,_ = engine.generate_batch(tokens, study_embeddings=sample_embeddings ,num_samples=1, max_tokens=1024, temperature=0.0, seed=10)
output = tokenizer.decode(sample[0])
print("EXAMPLE REPORT:")
print(output)


# Create dataloaders
manifest=pd.read_csv("assets/manifest.csv")
reports=pd.read_pickle("assets/reports.pkl")
embeddings=torch.load("assets/embeddings.pt").unsqueeze(1)
with open("assets/token_bytes.pt", "rb") as f:
    token_bytes = torch.load(f, map_location=device)
pad_id = tokenizer.encode("<|pad|>")[0]
test_dataset = dataset.EchoReportDataset(manifest,reports,embeddings,tokenizer,'test')

from echollm.loss_eval import evaluate_core
scores = evaluate_core(engine, test_dataset)
print(f"Scores {scores}")
core_metric = np.nanmean(list(scores.values()))
print0(f"core: {core_metric:.2f}")
