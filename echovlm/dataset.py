import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

class EchoReportDataset(Dataset):
    # Shared embedding cache across all dataset instances
    _embedding_cache = {}

    def __init__(self, manifest, reports, embeddings, tokenizer, split, max_len=1024):
        self.manifest = manifest[manifest['split'] == split]

        self.reports = reports
        # Note: make sure embeddings are normalized per sample
        self.embeddings = embeddings
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self,idx):
        study_uid = self.manifest.iloc[idx]['study']
        text = self.reports[study_uid]
        study_embedding = self.embeddings[study_uid]
        ids = self.tok.encode(text)
        # prepend bos and append eos token
        bos_token_id = self.tok.encode_special("<|bos|>")
        eos_token_id = self.tok.encode_special("<|eos|>")
        ids = [bos_token_id] + ids + [eos_token_id]
        ids = ids[: self.max_len]
        ids = torch.tensor(ids, dtype=torch.long)
        x = ids[:-1]
        y = ids[1:]
        return {"input_ids": x, "labels": y, "study_embeddings":study_embedding, "study_uid":study_uid}
    
def collate_batch(batch, pad_id:int):

    max_len = max(x["input_ids"].numel() for x in batch)
    #max_len = 512
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    max_num_videos = max(x['study_embeddings'].shape[0] for x in batch)
    study_embeddings = torch.zeros((len(batch),max_num_videos, 512),dtype=torch.float32)
    study_mask   = torch.zeros((len(batch), max_num_videos), dtype=torch.bool)
    study_ids = []
    for i, ex in enumerate(batch):
        n = ex["input_ids"].numel()
        input_ids[i, :n] = ex["input_ids"]
        labels[i, :n] = ex["labels"]
        attn_mask[i, :n] = True

        n_vids = ex['study_embeddings'].shape[0]
        study_embeddings[i, :n_vids] = ex['study_embeddings']
        study_mask[i, :n_vids]  = True

        study_ids.append(ex['study_uid'])
    return {"input_ids":input_ids,
            "labels":labels, 
            "attention_mask":attn_mask,
            "study_embeddings": study_embeddings,
            "study_mask": study_mask,
            "study_uids":study_ids}
