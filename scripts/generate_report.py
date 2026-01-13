
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
from contextlib import contextmanager
import sys


# Add EchoPrime to path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "EchoPrime"))
from echo_prime import EchoPrime
ep = EchoPrime()

INPUT="my_heart"
stack_of_videos = ep.process_mp4s(INPUT)
encoded_study=ep.encode_study(stack_of_videos, visualize=False)
encoded_study = torch.nn.functional.normalize(torch.mean(encoded_study[:,:512],0),dim=0).cpu()
my_study = ep.candidate_embeddings[torch.argmax(encoded_study@ep.candidate_embeddings.T)].unsqueeze(0)

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
sample,_ = engine.generate_batch(tokens, study_embeddings=my_study ,num_samples=1, max_tokens=1024, temperature=0.0, seed=10)
output = tokenizer.decode(sample[0])
print("EXAMPLE REPORT:")
print(output)

