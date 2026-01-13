import pandas as pd
import json
import functools
import torch
import random
import pickle

# Load data like in EchoPrime code
with open("EchoPrime/assets/all_phr.json", encoding="utf-8") as f:
    all_phrases = json.load(f)

t_list = {k: [all_phrases[k][j] for j in all_phrases[k]] 
            for k in all_phrases}

phrases_per_section_list = {k: functools.reduce(lambda a,b: a+b, v) for (k,v) in t_list.items()}
phrases_per_section_list_org = {k: functools.reduce(lambda a,b: a+b, v) for (k,v) in t_list.items()}

def phrase_decode(phrase_ids):
    report = ""
    current_section = -1
    for sec_idx, phrase_idx, value in phrase_ids:
        section=list(phrases_per_section_list_org.keys())[sec_idx]
        if sec_idx!=current_section:
            if current_section!=-1:
                report+="\n"
            report += section + ": "
            current_section=sec_idx


        phr = phrases_per_section_list_org[section][phrase_idx]

        if '<numerical>' in phr:
            phr = phr.replace('<numerical>',str(value))
        elif '<string>' in phr:
            phr = phr.replace('<string>',str(value))
            
        report += phr + " "
    # remove empty space
    report=report[:-1]
    return report

candidate_reports = pd.read_pickle("EchoPrime/model_data/candidates_data/candidate_reports.pkl")
reports = [phrase_decode(vec_phr) for vec_phr in candidate_reports]
studies=list(pd.read_csv("EchoPrime/model_data/candidates_data/candidate_studies.csv")['Study'])
embeddings_p1=torch.load("EchoPrime/model_data/candidates_data/candidate_embeddings_p1.pt")
embeddings_p2=torch.load("EchoPrime/model_data/candidates_data/candidate_embeddings_p2.pt")
embeddings=torch.cat((embeddings_p1,embeddings_p2),dim=0)
embeddings = torch.nn.functional.normalize(embeddings, dim=1)
# -----------------------------------------------------------------------------


# Create a manifest
split = ['train']*(len(studies)-20_000) + ['val']*10_000 + ['test']*10_000
random.seed(42)
random.shuffle(split)
manifest = pd.DataFrame({"study":studies, "split":split})
# -----------------------------------------------------------------------------

# Save to assets
manifest.to_csv("assets/manifest.csv",index=0)
with open("assets/reports.pkl",'wb') as f:
    pickle.dump(reports,f)
torch.save(embeddings,"assets/embeddings.pt")