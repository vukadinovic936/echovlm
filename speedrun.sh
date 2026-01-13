#!/bin/bash

# Python venv setup with uv
echo "Setting up the environment ..."
# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
uv add --editable .
echo "Finished setting up the environment"
# -----------------------------------------------------------------------------

# Download EchoPrime repository, we'll use this to get syntehtic training data
echo "Downloading EchoPrime"
git clone https://github.com/echonet/EchoPrime
cd EchoPrime
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/candidate_embeddings_p1.pt
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/candidate_embeddings_p2.pt
unzip model_data.zip
mv candidate_embeddings_p1.pt model_data/candidates_data/
mv candidate_embeddings_p2.pt model_data/candidates_data/
cd ..
cp -r EchoPrime/assets/ .
cp -r EchoPrime/model_data/ .
echo "Finished downloading EchoPrime"
# -----------------------------------------------------------------------------

# Prepare the dataset
echo "Preparing the dataset"
python -m scripts.prepare_dataset
echo "Finished preparing the dataset"
# -----------------------------------------------------------------------------

# Train and evaluate the tokenizer
echo "Training a tokenizer"
python -m scripts.tok_train
python -m scripts.tok_eval
echo "Finished training a tokenizer"
# -----------------------------------------------------------------------------

# Train the VLM
echo "Started VLM training"
# Note I set 1 gpu here but it's implemented to run on many gpus and can be ran across computers too. 
# If you have more gpus just increase nproc_per_node and if you want to run multi-comp see this blog https://jacksoncakes.com/getting-started-with-distributed-data-parallel-in-pytorch-a-beginners-guide/
torchrun --standalone --nproc_per_node=1 -m scripts.base_train
echo "Finished VLM training"
# -----------------------------------------------------------------------------

# generate a report of a real scan
python -m scripts.generate_report
sleep infinity
