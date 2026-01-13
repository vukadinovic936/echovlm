import math
import torch
import torch.distributed as dist
import json 
import re
import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics
from tqdm import tqdm
import pickle
# read data/extended_label_phrases.json
with open('assets/per_section.json', "r") as f:
    raw_tasks = json.load(f)
sorted_features=['impella',
    'ejection_fraction',
    'pacemaker',
    'rv_systolic_function_depressed',
    'right_ventricle_dilation',
    'left_atrium_dilation',
    'right_atrium_dilation',
    'mitraclip',
    'mitral_annular_calcification',
    'mitral_stenosis',
    'mitral_regurgitation',
    'tavr',
    'bicuspid_aov_morphology',
    'aortic_stenosis',
    'aortic_regurgitation',
    'tricuspid_stenosis',
    'tricuspid_valve_regurgitation',
    'pericardial_effusion',
    'aortic_root_dilation',
    'dilated_ivc',
    'pulmonary_artery_pressure_continuous']
tasks = {k:raw_tasks[k] for k in raw_tasks}

@torch.no_grad()
def evaluate_core(engine, dataset):
    """
    Evaluate accuracy of predicting core metrics
    """
    preds = {}
    gt={}
    prompt=""
    tokens = engine.tokenizer(prompt, prepend="<|bos|>")
    for idx,sample in tqdm(enumerate(dataset),total=len(dataset)):
        report,_ = engine.generate_batch(tokens,
                                     study_embeddings=sample['study_embeddings'],
                                     num_samples=1, 
                                     max_tokens=1024,
                                     temperature=0.0)
        report = engine.tokenizer.decode(report[0])
        preds[sample['study_uid']] = get_labels(report)
        ids = sample['input_ids']
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        real_report=engine.tokenizer.decode(ids)
        gt[sample['study_uid']] = get_labels(real_report)
            
    val_studies = list(preds.keys())

    # now we have a dictionary of ground truth labels in gt
    # and we have a dictionary of predictions in preds
    scores=[]
    for task in tasks:
        p_idx = list(tasks.keys()).index(task)
        g = np.array([gt[study][p_idx] for study in val_studies])
        p = np.array([preds[study][p_idx] for study in val_studies])
        mask = np.array([(gi != -1 and pi != -1) for gi, pi in zip(g, p)])
        g_clean = g[mask]
        p_clean = p[mask]

        if tasks[task]['mode']=='binary':
            scores.append(round(sklearn.metrics.roc_auc_score(g_clean,p_clean),2))
        else:
            mask = np.array([(pi<200 and gi<200) for gi, pi in zip(g_clean, p_clean)])
            g_clean = g_clean[mask]
            p_clean = p_clean[mask]
            #regression
            scores.append(round(sklearn.metrics.r2_score(g_clean,p_clean),2))
    # return a dictionary tasks/scores
    return {t:s for t,s in zip(tasks,scores)}

@torch.no_grad()
def evaluate_bpb(model, loader, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-indepedent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and indepependently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -100) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    for batch in loader:
        x = batch['input_ids'].to(model.get_device())
        y = batch['labels'].to(model.get_device())
        attn_mask = batch['attention_mask'].to(model.get_device())
        study_embeddings = batch['study_embeddings'].to(model.get_device())
        study_mask = batch['study_mask'].to(model.get_device())
        loss2d = model(x, y, attn_mask, study_embeddings, study_mask, loss_reduction='none') # (B,T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index(e.g. -100)
            # any target token <0 is to be ignored: do NOT index token_bytes with negatives
            valid = y>=0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes 
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes +=num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb           


def get_labels(report:str) -> list:
    """
    Given a report as text convert it to a list of labels
    """
    labels=[]
    for phenotype in list(tasks.keys()):
        lab=-1
        # recover list
        mode = tasks[phenotype]['mode']
        if mode=='regression':
            for phrase in tasks[phenotype]['label_sources']:
                pattern = re.compile((phrase.split("<#>")[0] + r"(\d{1,3}(?:\.\d{1,2})?)"), re.IGNORECASE)
                match = pattern.search(report)
                if match:
                    lab=float(match.group(1))
                    break
        elif mode=='binary':
            for phrase in tasks[phenotype]['label_sources']:
                if isin(phrase,report):
                    lab=1
                    break
            # positive phrase wasn't found then negative.
            if lab==-1:
                lab=0
        labels.append(lab)
    return labels

def isin(phrase,text):
    text = text.replace("\n", "")
    return phrase.lower() in (text.lower())