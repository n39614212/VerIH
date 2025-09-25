## Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```
## Data Generation
Setup you OpenAI Key
```
cd VerIH
python dataset/generate_ifeval.py
python examples/data_preprocess/make_parquet.py --dataset=iheval
```

## Training on Cluster
Set your own model ckpt in res/run.slurm
"export BASE_MODEL=/xxxx/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a"
```
cd res
sbath run.slurm
```
