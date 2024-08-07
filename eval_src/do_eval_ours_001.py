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
from collections import defaultdict

def extract_trace(data_item):
    res = []
    tot_res = []
    subanswer, tot, direct_answer = 0, 0, 0
    subanswer_directanswer, tot_directanswer, subanswer_tot, subanswer_tot_directanswer = 0, 0, 0, 0
    for item in data_item:
        i = 0
        trace = item["trace"] if "trace" in item else item
        while str(i) in trace:
            i += 1
        if "direct_answer" in trace[str(i-1)]:
            # res.append(trace[str(i-1)]["direct_answer"]["text"])
            if trace[str(i-1)]["tot_step"] != {}:
                if i != 1:
                    subanswer_tot_directanswer += 1
                    # tot_res.append(trace[str(i-1)]["direct_answer"]["text"])
                    if "answer is" in trace[str(i-1)]["direct_answer"]["text"]: res.append(trace[str(i-1)]["direct_answer"]["text"])
                else:
                    tot_directanswer += 1
                    # tot_res.append(trace[str(i-1)]["direct_answer"]["text"])
                    if "answer is" in trace[str(i-1)]["direct_answer"]["text"]: res.append(trace[str(i-1)]["direct_answer"]["text"])
            else:
                if i != 1:
                    subanswer_directanswer += 1
                    if "answer is" in trace[str(i-1)]["direct_answer"]["text"]: res.append(trace[str(i-1)]["direct_answer"]["text"])
                else:
                    direct_answer += 1
                    if "answer is" in trace[str(i-1)]["direct_answer"]["text"]: res.append(trace[str(i-1)]["direct_answer"]["text"])
        elif trace[str(i-1)]["tot_step"] != {}:
            j = 1
            while str(j) in trace[str(i-1)]["tot_step"]:
                j += 1
            # res.append(trace[str(i-1)]["tot_step"][str(j-1)])
            
            if i == 1: # only tot
                tot += 1
                # tot_res.append(trace[str(i-1)]["tot_step"][str(j-1)])
                if "answer is" in trace[str(i-1)]["tot_step"][str(j-1)]: res.append(trace[str(i-1)]["tot_step"][str(j-1)])
            else:
                subanswer_tot += 1
                # tot_res.append(trace[str(i-1)]["tot_step"][str(j-1)])
                if "answer is" in trace[str(i-1)]["tot_step"][str(j-1)]: res.append(trace[str(i-1)]["tot_step"][str(j-1)])
        elif "subanswer" in trace[str(i-1)]:
            if "answer is" in trace[str(i-1)]["subanswer"]["text"]: res.append(trace[str(i-1)]["subanswer"]["text"])
            subanswer += 1
        else:
            import pdb; pdb.set_trace()
    detail_count = [subanswer, tot, direct_answer, subanswer_directanswer, tot_directanswer, subanswer_tot, subanswer_tot_directanswer]
    assert sum(detail_count) == len(data_item)
    return res, detail_count

def extrace_completions(data_item):
    res = []
    for item in data_item:
        res.append(data_item[item]["model_solution"])
    return res

def eval_single_item_from_answer_sheets(example_id, answer_sheets_dir, evaluator: Evaluator):
    data_item = {}
    gold_answer = read_json(os.path.join(answer_sheets_dir, f"{example_id}.json"))["gold_answer"]
    data_1 = read_json(os.path.join(answer_sheets_dir, example_id.replace(" - Answer", " - Final Solutions") + ".json"))
    # data_2 = read_json(os.path.join(answer_sheets_dir, example_id.replace(" - Answer", " - Rollout Solutions") + ".json"))
    
    data_1, count_1 = extract_trace(data_1) # data_1 + data_2
    # model_answer_1, _, _, _ = evaluator.find_most_confident_answer(data_1)
    # result_1 = evaluator.check_answers_equiv(model_answer_1, gold_answer)
    # data_item["correctness_1"] = result_1
    # data_item["count_1"] = count_1

    # data_2, count_2 = extract_trace(data_2) # data_1 + data_2
    # model_answer_2, _, _, _ = evaluator.find_most_confident_answer(data_2)
    # result_2 = evaluator.check_answers_equiv(model_answer_2, gold_answer)
    # data_item["correctness_2"] = result_2
    # data_item["count_2"] = count_2

    # data_1_2 = data_1 + data_2
    # count_1_2 = [item_1 + item_2 for (item_1, item_2) in zip (count_1, count_2)]
    # model_answer_1_2, _, _, _ = evaluator.find_most_confident_answer(data_1_2)
    # result_1_2 = evaluator.check_answers_equiv(model_answer_1_2, gold_answer)
    # data_item["correctness_1_2"] = result_1_2
    # data_item["count_1_2"] = count_1_2

    return len(data_1)


def eval_exp(dataset_name: str):
    answer_sheets_dir = os.path.join(
        ".",
        "answer_sheets"
    )

    example_ids = [f.replace(".json", "") for f in os.listdir(answer_sheets_dir) \
                   if f.endswith(" - Answer.json") and \
                    True]
                    # int(f.replace(" - Answer.json", "").replace("Question ", "")) >19 and \
                    #     int(f.replace(" - Answer.json", "").replace("Question ", "")) <339 ]
    evaluator = eval(f"{dataset_name}Evaluator()")

    data_list = {}
    for example_id in tqdm(example_ids):
        cn = eval_single_item_from_answer_sheets(example_id, answer_sheets_dir, evaluator)
        data_list[int(example_id.replace("Question ", "").replace(" - Answer", ""))] = cn
    import pdb; pdb.set_trace()
        # except:
        #     print(example_id)
    # data_list = [ for example_id in tqdm(example_ids)]

    # # Calculate accuracy
    # print("Solution Node")
    # accuracy_1 = sum([item["correctness_1"] for item in data_list]) / len(data_list)
    # count_1 = [round(sum([item["count_1"][i] for item in data_list]) / len(data_list), 2) for i in range(len(data_list[0]["count_1"]))]
    # print(f"accuracy: {accuracy_1}, count: {count_1}")

    # print("Solution Node + Terminal Node")
    # accuracy_1_2 = sum([item["correctness_1_2"] for item in data_list]) / len(data_list)
    # count_1_2 = [round(sum([item["count_1_2"][i] for item in data_list]) / len(data_list), 2) for i in range(len(data_list[0]["count_1"]))]
    # print(f"accuracy: {accuracy_1_2}, count: {count_1_2}")

    # print("Terminal Node")
    # accuracy_2 = sum([item["correctness_2"] for item in data_list]) / len(data_list)
    # count_2 = [round(sum([item["count_2"][i] for item in data_list]) / len(data_list), 2) for i in range(len(data_list[0]["count_1"]))]
    # print(f"accuracy: {accuracy_2}, count: {count_2}")


    # # Save eval results
    # eval_result_dir = answer_sheets_dir.replace("run", "eval").replace("answer_sheets", "")
    # os.makedirs(eval_result_dir, exist_ok=True)
    # save_json(data_list, os.path.join(eval_result_dir, "eval_results_new.json"))
    # analysis = {"accuracy": accuracy}
    # save_json(analysis, os.path.join(eval_result_dir, "analysis_new.json"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    eval_exp(args.dataset_name)
