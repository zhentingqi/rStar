import sys; sys.path.append(".")
from common.helpers import read_json, save_json
from eval_src.Evaluator import Evaluator, GSM8KEvaluator, MATHEvaluator, FOLIOEvaluator, LOGIQAEvaluator, BGQAEvaluator

import os, multiprocessing as mp
import warnings; warnings.filterwarnings("ignore")
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from argparse import ArgumentParser


def eval_single_item(data_item, evaluator: Evaluator, vote_pool_dir: str, shot: int = None):
    try:
        pred_list = read_json(os.path.join(vote_pool_dir, data_item))
        gold_answer = read_json(os.path.join(vote_pool_dir.replace("vote_pool", "answer_sheets"), data_item))["gold_answer"]

        real_shot = len(pred_list) if shot == None else shot
        most_confident_answer, _, _, _ = evaluator.find_most_confident_answer(pred_list[:real_shot])
        t_corr = evaluator.check_answers_equiv(most_confident_answer, gold_answer)
    except Exception as e:
        print(f"Error: {e}")
        t_corr = 0
    return t_corr


def eval_vote_pool(vote_pool_dir: str, dataset_name: str, shot: int = None):
    """
    Evaluate the generated response from vote_pool. Only support self-consistency few-shot.

    Parameters: 
        vote_pool_dir: The path to the generated response. 
        save_path: The path to save the results. The default save_path is "evaluation/{data_file_name}"
        save_result: bool. If True, save the results. 
    """
    data_list = [js_file for js_file in os.listdir(vote_pool_dir) if js_file.endswith(".json")]    
    # if len(data_list) != 450:
    #     print(f"Warning: The number of data items is not 450 but {len(data_list)}.")
    #     exit()
    evaluator = eval(f"{dataset_name}Evaluator()")

    with Pool(mp.cpu_count()) as p:
        correctness_list = list(tqdm(p.map(
            partial(eval_single_item, evaluator=evaluator, vote_pool_dir=vote_pool_dir, shot=shot),
            data_list
        ), total=len(data_list)))
        
    # Calculate accuracy
    accuracy = sum(correctness_list) / len(correctness_list)
    # if len(correctness_list) == 450:
    #     accuracy = accuracy * 450 / 500
    #     print("base 500")
    print(f"For {vote_pool_dir}, maj@{shot} accuracy: {accuracy:.4f}")
    
    # Save eval results    
    eval_result_dir = vote_pool_dir.replace("run", "eval").replace("vote_pool", "")
    os.makedirs(eval_result_dir, exist_ok=True)
    analysis = {f"maj@{shot}accuracy": accuracy}
    save_json(analysis, os.path.join(eval_result_dir, "maj@{shot}_analysis.json"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--vote_pool_dir_path", type=str, required=True)
    parser.add_argument("--shot", type=int, choices=[8, 64])
    args = parser.parse_args()
    
    eval_vote_pool(args.vote_pool_dir_path, args.dataset_name, args.shot)
