from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from echovlm import dataset
from echovlm import tokenizer
import pandas as pd
import torch
from echovlm.gpt import GPTConfig,GPT
from echovlm.common import compute_init,autodetect_device_type,print0,compute_cleanup
from echovlm.loss_eval import evaluate_bpb
from contextlib import nullcontext
import csv
import os
import time
from echovlm.engine import Engine
from echovlm.plotting import save_plot
import numpy as np

# Compute Init
base_dir = "assets/"
device_type=""
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# -----------------------------------------------------------------------------
# User settings
run = "echovlm"
if master_process:
    dir_name=run
    i=1
    while os.path.exists(f"assets/{dir_name}"):
        dir_name = f"{run}{i}"
        i+=1
    run_dir = os.path.join(base_dir,dir_name)
    os.mkdir(run_dir)

max_seq_len=1024
avg_tokens_per_report=500 # since sequence length is variable, we'll take approx the average token length instead of calculating it every iter TODO: check if there's a better way
vocab_size=2000
num_layers=8
model_dim=512
num_heads=max(1, (model_dim+127)//128)
num_kv_heads=num_heads
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
grad_clip = 1.0
device_batch_size = 128
total_batch_size = avg_tokens_per_report * device_batch_size * ddp_world_size # how many tokens in total are processed per batch (approximately because tokens vary per sequence)
# ReduceLROnPlateau settings
lr_reduce_factor = 0.5 # factor to reduce LR by
lr_patience = 3 # number of evals without improvement before reducing LR
lr_threshold = 1e-4 # minimum change to qualify as an improvement
max_lr_reductions = 4 # stop training after this many LR reductions
eval_every=250
sample_every=1_000_000_000 # calculating core is slow do that at the end.
# -----------------------------------------------------------------------------

# CSV Logging Init
if master_process:
    train_csv_path = os.path.join(run_dir, "train.csv")
    val_csv_path = os.path.join(run_dir, "val.csv")
    with open(train_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total_training_flops", "total_training_time", "loss", "lr", "dt", "tok_per_sec"])
    with open(val_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total_training_flops", "total_training_time", "bpb", "core"])

# Plot tracking lists
plot_train_steps, plot_train_loss = [], []
plot_val_steps, plot_val_bpb = [], []
plot_core_steps, plot_core_metric = [], []
# -----------------------------------------------------------------------------

# Create dataloaders
manifest=pd.read_csv("assets/manifest.csv")
reports=pd.read_pickle("assets/reports.pkl")
embeddings=torch.load("assets/embeddings.pt").unsqueeze(1)
tokenizer = tokenizer.RustBPETokenizer.from_directory("assets/tokenizer")
with open("assets/token_bytes.pt", "rb") as f:
    token_bytes = torch.load(f, map_location=device)
pad_id = tokenizer.encode("<|pad|>")[0]
datasets = {split:dataset.EchoReportDataset(manifest,reports,embeddings,tokenizer,split)
                for split in ['train','val','test']}
samplers = {split: DistributedSampler(ds, shuffle=True) if ddp else None for split, ds in datasets.items()}
dataloaders = {split: DataLoader(ds,
                                 batch_size=device_batch_size,
                                 shuffle=(samplers[split] is None),
                                 sampler=samplers[split],
                                 num_workers=8,
                                 pin_memory=True,
                                 collate_fn=lambda b: dataset.collate_batch(b, pad_id)) for split,ds in datasets.items()}
# -----------------------------------------------------------------------------

# Initialize the Model and try a forward pass
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
#this initializes a meta model, weights don't occupy the GPU
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
#copied storage but not acutal values
model.to_empty(device=device)
#only now we are initializing weights
model.init_weights()

# Note: No DDP wrapper needed - gradient sync is handled by DistAdamW/DistMuon optimizers (ZeRO-2 style)

# compiles for efficiency, comment out if debugging, our sequences are not static so a bit irrelevant?TODO:check if it make sense to pad all to 1024?
# model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")
# -----------------------------------------------------------------------------

# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers
# -----------------------------------------------------------------------------

# Set up hyperparameter schedulers
# ReduceLROnPlateau for both optimizers
schedulers = [
    ReduceLROnPlateau(opt, mode='min', factor=lr_reduce_factor, patience=lr_patience, threshold=lr_threshold)
    for opt in optimizers
]
lr_reduction_count = 0
prev_lr = optimizers[0].param_groups[0]['lr']

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum
# -----------------------------------------------------------------------------

# Training loop
# calculate bits per byte which is easier to interpret than perplexity (baseline=8)
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
train_it = iter(dataloaders['train'])
step = 0
stop_training = False
while not stop_training:
    flops_so_far = num_flops_per_token * total_batch_size * step
    # once in a while: evaluate the val bpb (all ranks participate)
    if step % eval_every == 0:
        model.eval()
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, dataloaders['val'], token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f} | LR reductions: {lr_reduction_count}/{max_lr_reductions}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        if master_process:
            with open(val_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, flops_so_far, total_training_time, val_bpb, ""])
            plot_val_steps.append(step)
            plot_val_bpb.append(val_bpb)
            save_plot(plot_val_steps[1:], plot_val_bpb[1:], 'Validation BPB', 'val_bpb.png', run_dir)
        # Step the schedulers and check if LR was reduced
        for sched in schedulers:
            sched.step(val_bpb)
        curr_lr = optimizers[0].param_groups[0]['lr']
        if curr_lr < prev_lr:
            lr_reduction_count += 1
            print0(f"ReduceLROnPlateau: LR reduced to {curr_lr:.6f}. Reduction count: {lr_reduction_count}/{max_lr_reductions}")
            prev_lr = curr_lr
            if lr_reduction_count >= max_lr_reductions:
                print0(f"ReduceLROnPlateau: reached {max_lr_reductions} LR reductions. Stopping training.")
                stop_training = True
        model.train()

    # once in a while: sample from the model (only on the master process)
    if master_process and (stop_training or (step > 0 and step % sample_every == 0)):
        model.eval()

        prompts = [
            "A complete echo was performed using 2D, spectral Doppler, and",
            "Normal systolic function with an estimated EF of 60 - 65%.",
            "Pulmonic Valve\n Pulmonic valve ",
            "Mitral Valve\n Structurally normal mitral valve.",
        ]
        engine = Engine(model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample,_ = engine.generate_batch(tokens, num_samples=1, max_tokens=1024, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint at the end of the run (only on master process)
    if master_process and stop_training:
        checkpoint = {
            "step": step,
            "model": model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "val_bpb": val_bpb,
            "model_config": model_config_kwargs,
            "device_batch_size": device_batch_size,
            "max_seq_len": max_seq_len,
        }
        torch.save(checkpoint, os.path.join(run_dir, f"checkpoint{step:06d}.pt"))
    if stop_training:
        break


    # -------------------------------------------------------------------------
    # single training step
    # get a batch, reset iterator if epoch ends
    try:
        sample = next(train_it)
    except StopIteration:
        train_it = iter(dataloaders['train'])
        sample = next(train_it)
    x,y,mask,study_embeddings,study_mask = sample['input_ids'].to(device), sample['labels'].to(device), sample['attention_mask'].to(device), sample['study_embeddings'].to(device), sample['study_mask'].to(device)

    # evaluate the gradient
    # sync for time purpose
    synchronize()
    t0 = time.time()
    with autocast_ctx:
        loss = model(x, y, mask, study_embeddings, study_mask)
    train_loss = loss.detach()
    loss.backward()
    # gradient clipping
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer (LR is managed by reduce_lr_on_plateau)
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step+1)) # debias the EMA bc we started smooth_train_loss at 0
    tok_per_sec = int( (total_batch_size) / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    current_lr = optimizers[0].param_groups[0]['lr']
    print0(f"step {step:05d} | loss: {debiased_smooth_loss:.6f} | lr: {current_lr:.6f} | dt: {dt* 1000:.2f}ms | tok/sec: {tok_per_sec:,} | total time: {total_training_time/60:.2f}m")
    if master_process and step % 100 == 0:
        with open(train_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, flops_so_far, total_training_time, debiased_smooth_loss, current_lr, dt, tok_per_sec])
        plot_train_steps.append(step)
        plot_train_loss.append(debiased_smooth_loss)
        save_plot(plot_train_steps[1:], plot_train_loss[1:], 'Training Loss', 'train_loss.png', run_dir)
    step += 1
print0(f"Peak memory usage {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

compute_cleanup()
