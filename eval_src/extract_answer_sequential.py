import sys

sys.path.append(".")
from common.helpers import read_json, save_json
from eval_src.Evaluator import Evaluator, GSM8KEvaluator, MATHEvaluator, FOLIOEvaluator, LOGIQAEvaluator, BGQAEvaluator

import os, multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from argparse import ArgumentParser
from collections import defaultdict, Counter

def most_common_element(lst):
    if len(lst) == 0:
        return None
    count = Counter(lst)
    most_common = count.most_common(1)[0]
    return most_common[0]


def extract_trace(data_item, evaluator):
    res = defaultdict(list)
    for item in data_item:
        i = 0
        trace = item["trace"] if "trace" in item else item
        if "rollout_id" not in item:
            item["rollout_id"] = 0
        while str(i) in trace:
            i += 1
        if "direct_answer" in trace[str(i-1)]:
            res[item["rollout_id"]].append(evaluator.extract_answer_from_model_completion(trace[str(i-1)]["direct_answer"]["text"]))
        elif len(trace[str(i-1)]["tot_step"]) != 0:
            j = 1
            while str(j) in trace[str(i-1)]["tot_step"]:
                j += 1
            res[item["rollout_id"]].append(evaluator.extract_answer_from_model_completion(trace[str(i-1)]["tot_step"][str(j-1)]))
        elif "subanswer" in trace[str(i-1)]:
            res[item["rollout_id"]].append(evaluator.extract_answer_from_model_completion(trace[str(i-1)]["subanswer"]["text"]))
        
        else:
            import pdb; pdb.set_trace()
    return res


def eval_single_item_from_answer_sheets(example_id, answer_sheets_dir, evaluator: Evaluator):
    gold_answer = read_json(os.path.join(answer_sheets_dir, f"{example_id}.json"))["gold_answer"]
    data_1 = read_json(os.path.join(answer_sheets_dir, example_id.replace(" - Answer", " - Final Solutions") + ".json"))
    try: data_2 = read_json(os.path.join(answer_sheets_dir, example_id.replace(" - Answer", " - Rollout Solutions") + ".json"))
    except: pass

    data_1 = extract_trace(data_1, evaluator)
    try: data_2 = extract_trace(data_2, evaluator)
    except: pass
    answer_list = []
    for i in range(max(data_1.keys()) + 1):
        answer_list.extend(data_1[i])
        try: answer_list.append(data_2[i][0])
        except: pass

    model_limit_correct = False
    for item in answer_list:
        if evaluator.check_answers_equiv(item, gold_answer):
            model_limit_correct = True; break
    major_vote_correct = evaluator.check_answers_equiv(most_common_element(answer_list), gold_answer)
    
    data_item = {
        "original_id": int(example_id.replace(" - Answer", "").replace("Question ", "")),
        "ground_truth": gold_answer,
        "answer_list": answer_list,
        "model_limit_correct": model_limit_correct,
        "major_vote_correct": major_vote_correct
    }
        
    save_json(data_item, os.path.join(
        answer_sheets_dir,
        example_id.replace(" - Answer", " - Sequential List") + ".json"))
    return data_item


def eval_exp(dataset_name: str):
    answer_sheets_dir = os.path.join(args.exp_dir_path, "answer_sheets")

    example_ids = [f.replace(".json", "") for f in os.listdir(answer_sheets_dir) if f.endswith(" - Answer.json") and "200" not in f]
    evaluator = eval(f"{dataset_name}Evaluator()")

    # with Pool(mp.cpu_count()) as p:
    #     data_list = list(
    #         tqdm(
    #             p.imap_unordered(
    #                 partial(
    #                     eval_single_item_from_answer_sheets,
    #                     answer_sheets_dir=answer_sheets_dir,
    #                     evaluator=evaluator,
    #                     vote_pool_dir=vote_pool_dir,
    #                     num_votes=num_votes,
    #                 ),
    #                 example_ids,
    #             ),
    #             total=len(example_ids),
    #         )
    #     )

    data_list = [eval_single_item_from_answer_sheets(example_id, answer_sheets_dir, evaluator) for example_id in tqdm(example_ids)]

    # Calculate accuracy
    accuracy = sum([item["major_vote_correct"] for item in data_list]) / len(data_list)
    print(f"For {answer_sheets_dir}, accuracy: {accuracy}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--exp_dir_path", type=str, required=True)
    args = parser.parse_args()

    eval_exp(args.dataset_name)
