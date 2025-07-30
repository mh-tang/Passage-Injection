USER_PROMPT = "You are given several passages and a question. Please answer the question concisely, following the format shown in the example below.\n\
Example:\n\
Question: What is the boiling point of water?\n\
Your Response: The answer is 100°C.\n\n\
Passages:\n\
{passages}\n\n\
Using both the information from the passages and your own knowledge, answer the following question in the same format. Do not include any additional explanation.\n\
Question: {question}"


USER_PROMPT_WITHOUT_P = "You are given a question. Please answer the question concisely, following the format shown in the example below.\n\
Example:\n\
Question: What is the boiling point of water?\n\
Your Response: The answer is 100°C.\n\n\
Answer the following question in the same format. Do not include any additional explanation.\n\
Question: {question}"


USER_PROMPT_WITHOUT_INS = "You are given several passages and a question. Please answer the question concisely, following the format shown in the example below.\n\
Example:\n\
Question: What is the boiling point of water?\n\
Your Response: The answer is 100°C.\n\n\
Passages:\n\
{passages}\n\n\
Answer the following question in the same format. Do not include any additional explanation.\n\
Question: {question}"


def qwen3_template(instruction):
    data = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n'
    return data

    
def qwen3_passages_template(instruction, passages):
    data = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n<think>There are several passages:\n{passages}\n\nTo answer the question accurately and in the correct format, I should use both the information from the passages and my own knowledge.'
    return data


def qwen3_instruct_template(instruction):
    data = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n<think>To answer the question accurately and in the correct format, I should use both the information from the passages and my own knowledge.'
    return data


def deepseek_r1_distill_qwen_template(instruction):
    data = f"<｜begin▁of▁sentence｜><｜User｜>{instruction}<｜Assistant｜>"
    return data


def deepseek_r1_distill_qwen_passages_template(instruction, passages):
    data = f"<｜begin▁of▁sentence｜><｜User｜>{instruction}<｜Assistant｜><think>There are several passages:\n{passages}\n\nTo answer the question accurately and in the correct format, I should use both the information from the passages and my own knowledge."
    return data


def deepseek_r1_distill_qwen_instruct_template(instruction):
    data = f"<｜begin▁of▁sentence｜><｜User｜>{instruction}<｜Assistant｜><think>To answer the question accurately and in the correct format, I should use both the information from the passages and my own knowledge."
    return data