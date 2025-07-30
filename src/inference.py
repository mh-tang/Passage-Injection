import os
import argparse
from vllm import LLM, SamplingParams
from utils import load_data, save_output, format_nopassage_prompt, format_prompt, format_passage_prompt, format_instruction_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="Qwen3-32B")
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
    
    for filename, data in data_list:  # deal with each file
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        if 'confiqa' in filename:  # ignore confiqa dataset
            continue

        output_dir = f"./outputs/{model_name}/top{args.topk}/{filename}"
        os.makedirs(output_dir, exist_ok=True)

        # direct qa
        prompts = [
            format_nopassage_prompt(item["question"], type)
            for item in data
        ]
        outputs = llm.generate(prompts, sampling_params)
        output_path = f"{output_dir}/output_direct.json"
        save_output(outputs, output_path, data, prompts)

        # vanilla rag
        prompts = [
            format_prompt(item["passages"][:args.topk], item["question"], type)
            for item in data
        ]
        outputs = llm.generate(prompts, sampling_params)
        output_path = f"{output_dir}/output_rag.json"
        save_output(outputs, output_path, data, prompts)

        # passage injection
        prompts = [
            format_passage_prompt(item["passages"][:args.topk], item["question"], type)
            for item in data
        ]
        outputs = llm.generate(prompts, sampling_params)
        output_path = f"{output_dir}/output_passage.json"
        save_output(outputs, output_path, data, prompts)

        # instruction injection
        prompts = [
            format_instruction_prompt(item["passages"][:args.topk], item["question"], type)
            for item in data
        ]
        outputs = llm.generate(prompts, sampling_params)
        output_path = f"{output_dir}/output_instruction.json"
        save_output(outputs, output_path, data, prompts)


