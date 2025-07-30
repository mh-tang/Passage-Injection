# Passage Injection


Welcome to the Official Repository of Passage Injection!

This repository contains the code, datasets, and models used in our paper: [Injecting External Knowledge into the Reasoning Process Enhances Retrieval-Augmented Generation](https://arxiv.org/abs/2507.19333).

Passage Injection is a simple yet effective method that explicitly incorporates retrieved passages into LLMs' reasoning process to enhance robustness against noisy information and improve RAG performance.

![Overall Method](overall.png)


## Reproduce Paper Results

### Install Environment
```bash
conda create -n passage_injection python=3.11.2
conda activate passage_injection
pip install vllm==0.8.5
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### Download Datasets


### Retrieve Passages


### Run Passage Injection

```bash
python src/inference.py --model_name Qwen3-32B --topk 5

python src/evaluate.py --model_name Qwen3-32B --topk 5
```


```bash
python src/infer_further.py --model_name Qwen3-32B --further_type random_noise

python src/evaluate.py --model_name Qwen3-32B --further_type random_noise
```
