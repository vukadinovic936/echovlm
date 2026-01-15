# echovlm
> Medical Imaging Report Generation Model for $5

![echovlm demo](https://github.com/vukadinovic936/echovlm/blob/main/demo.gif)

This repo is a full-stack implementation of a VLM for medical report generation from imaging scans. It is designed to serve as a practical example for researchers, demonstrating end-to-end training of VLMs on medical imaging data, and can be adapted to various imaging modalities.
echovlm is inspired by Karpathy's [nanochat](https://github.com/karpathy/nanochat), and is also a fully open source, reproducible codebase which makes it one of the few public codebases for medical machine learning. As a running example, we use an echocardiography dataset with synthethic reports and study embeddings, which allows us to simulate VLM training data despite not having access to raw medical imaging scans. echovlm is very light and it can run on a single gpu via speedrun.sh script, that runs the entire pipeline start to end. This includes dataset preparation, tokenization, training, evaluation, inference and real example on the scan of my heart.


# Quick Start
```bash
bash speedrun.sh
```
Since the training takes some time, I run it in the background with [tmux](https://tmuxcheatsheet.com/how-to-install-tmux/)
```bash
tmux new-session -d -s speedrun 'bash speedrun.sh'
```
You can subsequently attach tmux with `tmux attach-session` and then detach with 'Ctrl+b :detach'
This bash script will run the whole pipeline end-to-end, if you want a more detauled walkthrough of around the speedrun script see ("echovlm: Medical Imaging Report Generation Model for $5")[https://github.com/vukadinovic936/echovlm/discussions/1]
# Evaluation 
In medical report generation, diagnostic accuracy is the metric we care about the most. We'll see how well can how well can echovlm assess common cardiac traits from study embeddings. Note that these study embeddings are not actually encoded videos, and are not directly comparable to SOTA echocardiography modes, but will work for the purposes of demonstration. You can run 
```bash
python -m scripts.test_inference
```
The output will include individual scores (AUROC for binary, r2 for regression) and a **Core Metric**, which represents the average across all evaluated traits.
### Model Performance Metrics

| Cardiac Trait / Finding | Score |
| :--- | :--- |
| pacemaker | 0.73 |
| impella | 0.5 |
| tavr | 0.94 |
| mitraclip | 0.91 |
| aortic_root_dilation | 0.61 |
| bicuspid_aov_morphology | 0.66 |
| aortic_stenosis | 0.77 |
| tricuspid_stenosis | 0.5 |
| aortic_regurgitation | 0.56 |
| dilated_ivc | 0.61 |
| left_atrium_dilation | 0.82 |
| ejection_fraction R2 | 0.68 |
| mitral_annular_calcification | 0.89 |
| mitral_stenosis | 0.56 |
| mitral_regurgitation | 0.7 |
| pericardial_effusion | 0.93 |
| pulmonary_artery_pressure_continuous R2 | 0.23 |
| right_atrium_dilation | 0.54 |
| rv_systolic_function_depressed | 0.67 |
| right_ventricle_dilation | 0.71 |
| tricuspid_valve_regurgitation | 0.78 |
| pulmonic_valve_regurgitation | 0.5 |
| elevated_left_atrial_pressure | 0.57 |
| wall_motion_hypokinesis | 0.75 |
| atrial_septum_hypertrophy | 0.5 |
| **Core Metric (Average)** | **0.66** |

# Contributing
echovlm can serve as both a starting point to train vlms on your own data and a learning example for medical vlm training. If you use it for the purposes of learning, and find yourself running this code, there are potentially many tweaks you can use to improve the speed, clarity or performance, I welcome new pull requests! Possible experiments are: transfer learning from an llm, sft for question answering, rlhf with grpo to learn preferences where diagnoses are accurate (see core metrics).

# Using echovlm for your research
If you are considering to use echovlm for your own research here are the advantages of using it versus general vlm training libraries:
1) Medical reports have much smaller vocabulary size than natural language, you'd benefit from training a custom tokenizer as shown here.
2) We significantly reduce training time and hardware requirements by using a previously developed encoder for echocardiography, so during vlm training, vlm sees only the embeddings. Given that there are so many publicly available encoders for almost all medical imaging modalities ([MRI](https://www.nature.com/articles/s41551-024-01283-7), [ECG](https://ai.nejm.org/doi/full/10.1056/AIoa2401033), [X-ray](https://www.nature.com/articles/s41586-025-09079-8) etc ...) this is a useful trick.
3) Consider that when using this code on your own data you'd have to find the optimal hyperparameters for your dataset such as the vocabulary size, maximum sequence length and number of transformer layers.

# Cite
```bibtex
@misc{echovlm,
  author = {Milos Vukadinovic},
  title = {echovlm: Medical Imaging Report Generation for X%},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/vukadinovic936/echovlm}
}
```

# License 
MIT