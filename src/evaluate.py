import os
import json
import re
import string
import argparse
import numpy as np
from collections import Counter
from typing import Union, List
from utils import load_outputs


class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric
    

def evaluate(pred, ground_truth):
    if "The answer is" in pred:
        pred = pred[pred.find("The answer is") + len("The answer is"):]
    pred = pred.strip()
    stop_list = [".", "\n", ","]
    for stop in stop_list:
        end_pos = pred.find(stop)
        if end_pos != -1:
            pred = pred[:end_pos].strip() 

    em = BaseDataset.exact_match_score(
        prediction=pred,
        ground_truth=ground_truth,
    )["correct"]
    f1_score = BaseDataset.f1_score(
        prediction=pred,
        ground_truth=ground_truth,
    )
    f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
    return {
        "eval_predict": pred,
        "em": str(em),
        "f1": str(f1),
        "prec": str(prec),
        "recall": str(recall),
    }


def main(args, datasetnames):
    model_name = args.model_name
    all_avg_results = {}  # avg results of every method
    total_cnt = {}

    for datasetname in datasetnames:
        print(f"### Evaluating {datasetname} ###")
        
        if args.further_type is None:
            all_outputs = load_outputs(f'./outputs/{model_name}/top{args.topk}/{datasetname}')
        else:
            all_outputs = load_outputs(f'./outputs/{model_name}/{args.further_type}/{datasetname}')
        
        for filename, output in all_outputs:
            filename = filename.split(".")[0]
            filename = filename.split("_")[1]
            print(f"### Solving {filename} ###")

            if filename not in all_avg_results:  # cal avg
                all_avg_results[filename] = {}
                total_cnt[filename] = 0

            results = []
            for item in output:
                question = item['question']
                answer = item['answer']
                pred = item['output_answer']
                ret = {
                    "question": question,
                    "answer": answer,
                    "pred": pred,
                }
                ret.update(evaluate(pred, answer))  # cal metrics
                results.append(ret)

            if args.further_type is None:
                out_dir_prefix = f'./results/{model_name}/top{args.topk}'
            else:
                out_dir_prefix = f'./results/{model_name}/{args.further_type}'
            out_dir = f'{out_dir_prefix}/{datasetname}/prediction'
            os.makedirs(out_dir, exist_ok=True)

            # save prediction
            with open(f'{out_dir}/prediction_{filename}.json', 'w') as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)

            # evaluation metrics
            metrics = ["em", "f1", "prec", "recall"]
            ret_str = ""
            for met in metrics:
                acc = sum(float(d[met]) for d in results) / len(results)
                acc = round(acc, 4)
                ret_str += f"{met}\t{acc}\n"

                if met not in all_avg_results[filename]:  # cal avg
                    all_avg_results[filename][met] = 0
                all_avg_results[filename][met] += sum(float(d[met]) for d in results)
                total_cnt[filename] += len(results) / len(metrics)
            
            with open(f'{out_dir_prefix}/{datasetname}/{filename}-metrics.txt', 'w') as fout:
                fout.write(ret_str)
            
    # save avg results
    with open(f'{out_dir_prefix}/avg-metrics.txt', 'w') as fout:
        for filename in all_avg_results:
            ret_str = f"=====file\t{filename}=====\n"
            print(filename, total_cnt[filename])
            for met in metrics:
                all_avg_results[filename][met] /= total_cnt[filename]  # cal avg
                acc = round(all_avg_results[filename][met], 4)
                ret_str += f"{met}\t{acc}\n"
            fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="Qwen3-32B")
    parser.add_argument("--further_type", type=str, default=None)
    args = parser.parse_args()
    
    if args.further_type == 'cf_noise':
        datasetnames = ['confiqa-qa', 'confiqa-mc', 'confiqa-mr']
    elif args.further_type == 'gold':
        datasetnames = ['2wiki-bridge', '2wiki-comparison', '2wiki-compositional', '2wiki-inference', 'hotpotqa-bridge', 'hotpotqa-comparison']
    else:
        datasetnames = ['2wiki-bridge', '2wiki-comparison', '2wiki-compositional', '2wiki-inference', 'cwq', 'hotpotqa-bridge', 'hotpotqa-comparison', 'popqa']
    
    main(args, datasetnames)
    