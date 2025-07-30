import os
import json
from prompt import *


def format_nopassage_prompt(question, type='qwen3'):
    instruction = USER_PROMPT_WITHOUT_P.format(question=question)
    if type == 'qwen3':
        return qwen3_template(instruction)
    else:
        return deepseek_r1_distill_qwen_template(instruction)


def format_prompt(passages, question, type='qwen3'):
    contexts = ""
    if passages is not None:
        for pid, psg in enumerate(passages):
            contexts += f"Passage {pid+1}: {psg}\n"

    instruction = USER_PROMPT.format(passages=contexts, question=question)
    if type == 'qwen3':
        return qwen3_template(instruction)
    else:
        return deepseek_r1_distill_qwen_template(instruction)


def format_passage_prompt(passages, question, type='qwen3'):
    contexts = ""
    if passages is not None:
        for pid, psg in enumerate(passages):
            contexts += f"Passage {pid+1}: {psg}\n"

    instruction = USER_PROMPT_WITHOUT_P.format(question=question)
    if type == 'qwen3':
        return qwen3_passages_template(instruction, contexts)
    else:
        return deepseek_r1_distill_qwen_passages_template(instruction, contexts)
    

def format_instruction_prompt(passages, question, type='qwen3'):
    contexts = ""
    if passages is not None:
        for pid, psg in enumerate(passages):
            contexts += f"Passage {pid+1}: {psg}\n"

    instruction = USER_PROMPT_WITHOUT_INS.format(passages=contexts, question=question)
    if type == 'qwen3':
        return qwen3_instruct_template(instruction)
    else:
        return deepseek_r1_distill_qwen_instruct_template(instruction)
    

def load_data(datadir):
    solve_dataset = []
    files = [f for f in os.listdir(datadir) if f.lower().endswith('.json')]
    
    for filename in files:
        with open(os.path.join(datadir, filename), "r") as fin:
            solve_dataset.append((filename, json.load(fin)))

    return solve_dataset


def save_output(outputs, output_path, data, prompts):
    out_dict = []
    for i, output in enumerate(outputs):
        full_out = output.outputs[0].text.strip()
        if "</think>" in full_out:
            out_answer = full_out.split("</think>")[1]
            out_answer = out_answer.lstrip("\n\n")
        else:
            out_answer = full_out
        
        out_dict.append({
            "prompt": prompts[i],
            "question": data[i]["question"],
            "answer": data[i]["answer"],
            "full_output": full_out,
            "output_answer": out_answer,
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=4, ensure_ascii=False)


def load_outputs(datadir):
    all_outputs = []
    files = [f for f in os.listdir(datadir)]
    files = [f for f in files if not os.path.isdir(os.path.join(datadir, f))]
    
    for filename in files:
        with open(os.path.join(datadir, filename), "r") as fin:
            all_outputs.append((filename, json.load(fin)))

    return all_outputs