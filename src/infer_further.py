import os
import random
import argparse
from vllm import LLM, SamplingParams
from utils import load_data, save_output, format_prompt, format_passage_prompt, format_instruction_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-32B")
    parser.add_argument("--further_type", type=str, default="random_noise", choices=["random_noise", "cf_noise", "gold"])
    args = parser.parse_args()

    model_name = args.model_name
    llm = LLM(
        model=f"./models/{model_name}",
        tokenizer=f"./models/{model_name}",
        dtype="bfloat16",
    )

    if 'Qwen3' in model_name:
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0,
            max_tokens=1024,
        )
        type = 'qwen3'
    else:
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=1024,
        )
        type = 'deepseek'

    data_list = load_data("./datasets")
    if args.further_type == 'cf_noise':
        datasetnames = ['confiqa-qa', 'confiqa-mc', 'confiqa-mr']
    elif args.further_type == 'gold':
        datasetnames = ['2wiki-bridge', '2wiki-comparison', '2wiki-compositional', '2wiki-inference', 'hotpotqa-bridge', 'hotpotqa-comparison']
    else:
        datasetnames = ['2wiki-bridge', '2wiki-comparison', '2wiki-compositional', '2wiki-inference', 'cwq', 'hotpotqa-bridge', 'hotpotqa-comparison', 'popqa']
    
    for filename, data in data_list:  # deal with each file
        filename = filename.split(".")[0]
        if filename not in datasetnames:
            continue
        print(f"### Solving {filename} ###")

        output_dir = f"./outputs/{model_name}/{args.further_type}/{filename}"
        os.makedirs(output_dir, exist_ok=True)

        # vanilla rag
        if args.further_type == 'random_noise':
            prompts = []
            for i, item in enumerate(data):
                other_indices = [j for j in range(len(data)) if j != i]
                random_index = random.choice(other_indices)
                noise_passages = data[random_index]["passages"][:3]

                prompt = format_prompt(noise_passages, item["question"], type)
                prompts.append(prompt)
        else:
            prompts = [
                format_prompt(item["golden_passages"], item["question"], type)
                for item in data
            ]
        outputs = llm.generate(prompts, sampling_params)
        output_path = f"{output_dir}/output_rag.json"
        save_output(outputs, output_path, data, prompts)

        # passage injection
        if args.further_type == 'random_noise':
            prompts = []
            for i, item in enumerate(data):
                other_indices = [j for j in range(len(data)) if j != i]
                random_index = random.choice(other_indices)
                noise_passages = data[random_index]["passages"][:3]

                prompt = format_passage_prompt(noise_passages, item["question"], type)
                prompts.append(prompt)
        else:
            prompts = [
                format_passage_prompt(item["golden_passages"], item["question"], type)
                for item in data
            ]
        outputs = llm.generate(prompts, sampling_params)
        output_path = f"{output_dir}/output_passage.json"
        save_output(outputs, output_path, data, prompts)

        # instruction injection
        if args.further_type == 'random_noise':
            prompts = []
            for i, item in enumerate(data):
                other_indices = [j for j in range(len(data)) if j != i]
                random_index = random.choice(other_indices)
                noise_passages = data[random_index]["passages"][:3]

                prompt = format_instruction_prompt(noise_passages, item["question"], type)
                prompts.append(prompt)
        else:
            prompts = [
                format_instruction_prompt(item["golden_passages"], item["question"], type)
                for item in data
            ]
        outputs = llm.generate(prompts, sampling_params)
        output_path = f"{output_dir}/output_instruction.json"
        save_output(outputs, output_path, data, prompts)


