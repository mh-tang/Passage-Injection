import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from retrieve.retriever import bm25_retrieve

random.seed(42)


def load_popqa(data_path):
    data_path = os.path.join(data_path, "popQA.tsv")
    dataset = pd.read_csv(data_path, sep="\t")
    new_dataset = []
    for did in range(len(dataset)):
        data = dataset.iloc[did]
        question = data["question"]
        answer = [data["obj"]] + eval(data["o_aliases"])
        val = {
            "test_id": did, 
            "question": question, 
            "answer": answer,
        }        
        new_dataset.append(val)
    return {"total": new_dataset}


def load_complexwebquestions(data_path):
    data_path = os.path.join(data_path, "ComplexWebQuestions_dev.json")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    new_dataset = []
    for did, data in enumerate(dataset):
        question = data["question"]
        answer = []
        for ans in data["answers"]:
            answer.append(ans["answer"])
            answer.extend(ans["aliases"])
        answer = list(set(answer))
        val = {
            "test_id": did, 
            "question": question, 
            "answer": answer,
        }        
        new_dataset.append(val)
    ret = {"total": new_dataset}
    return ret


def load_2wikimultihopqa(data_path):
    with open(os.path.join(data_path, "dev.json"), "r", encoding='utf-8') as fin:
        dataset = json.load(fin)
    with open(os.path.join(data_path, "id_aliases.json"), "r", encoding='utf-8') as fin:
        aliases = dict()
        for li in fin:
            t = json.loads(li)
            aliases[t["Q_id"]] = t["aliases"]
    new_dataset = []
    type_to_dataset = {}
    for did, data in enumerate(dataset):
        ans_id = data["answer_id"]
        val = {
            "qid": data["_id"], 
            "test_id": did, 
            "question": data["question"], 
            "answer": aliases[ans_id] if ans_id else data["answer"]
        }
        golden_passages = []
        contexts = {name: " ".join(sents) for name, sents in data["context"]}
        for fact_name, _sent_id in data["supporting_facts"]:
            psg = contexts[fact_name]
            golden_passages.append(psg)
        val["golden_passages"] = golden_passages
        val["type"] = data["type"]
        new_dataset.append(val)
        if data["type"] not in type_to_dataset:
            type_to_dataset[data["type"]] = []
        type_to_dataset[data["type"]].append(val)
    ret = {"total": new_dataset}
    ret.update(type_to_dataset)
    return ret


def load_hotpotqa(data_path):
    data_path = os.path.join(data_path, "hotpot_dev_distractor_v1.json")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    new_dataset = []
    type_to_dataset = {}
    for did, data in enumerate(dataset):
        val = {
            "qid": data["_id"], 
            "test_id": did, 
            "question": data["question"], 
            "answer": data["answer"]
        }
        tmp = []
        contexts = {name: "".join(sents) for name, sents in data["context"]}
        for fact_name, _sent_id in data["supporting_facts"]:
            psg = contexts[fact_name]
            tmp.append(psg)
        golden_passages = []
        for p in tmp:
            if p not in golden_passages:
                golden_passages.append(p)
        val["golden_passages"] = golden_passages
        val["type"] = data["type"]
        new_dataset.append(val)
        if data["type"] not in type_to_dataset:
            type_to_dataset[data["type"]] = []
        type_to_dataset[data["type"]].append(val)
    ret = {"total": new_dataset}
    ret.update(type_to_dataset)
    return ret


def load_default_format_data(data_path):
    filename = data_path.split("/")[-1]
    assert filename.endswith(".json"), f"Need json data: {data_path}"
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    for did, data in enumerate(dataset):
        assert "question" in data, f"\"question\" not in data, {data_path}"
        question = data["question"]
        assert type(question) == str, f"\"question\": {question} should be a string"
        assert "answer" in data, f"\"answer\" not in data, {data_path}"
        answer = data["answer"]
        assert type(answer) == str or \
               (type(answer) == list and (not any(type(a) != str for a in answer))), \
               f"\"answer\": {answer} should be a string or a list[str]" 
        data["test_id"] = did
    return {filename: dataset}


def main(args):
    output_dir ='datasets'
    os.makedirs(output_dir, exist_ok=True)

    print("### Loading dataset ###")
    if f"load_{args.dataset}" in globals():
        load_func = globals()[f"load_{args.dataset}"]
    else:
        load_func = globals()["load_default_format_data"]
    data_path = 'data/' + args.dataset

    load_dataset = load_func(data_path)
    if len(load_dataset) == 1:
        solve_dataset = load_dataset
    else:
        solve_dataset = {k: v for k, v in load_dataset.items() if k != "total"}
    
    for filename, dataset in solve_dataset.items():
        print(f"### Solving {filename} ###")
        base_filename = filename.split('_')[0] if '_' in filename else filename
        base_filename = '' if base_filename == "total" else f'-{base_filename}'
        output_name = f'{args.dataset}{base_filename}.json'
        output_file = os.path.join(
            output_dir, 
            output_name
        )
        ret = []
        pbar = tqdm(total = len(dataset) * args.topk)
        for data in dataset:
            passages = bm25_retrieve(data["question"], topk=args.topk)
            final_passages = []
            for psg in passages:
                final_passages.append(psg)
                pbar.update(1)
            data["passages"] = final_passages
            ret.append(data)
        with open(output_file, "w") as fout:
            json.dump(ret, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--topk", type=int, default=3) 
    args = parser.parse_args()
    print(args)
    main(args)