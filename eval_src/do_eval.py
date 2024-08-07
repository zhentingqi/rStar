import sys

sys.path.append(".")
from common.helpers import read_json, save_json
from eval_src.Evaluator import Evaluator, GSM8KEvaluator, MATHEvaluator, FOLIOEvaluator, LOGIQAEvaluator, BGQAEvaluator, MULTIARITHEvaluator

import os, multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from argparse import ArgumentParser


def eval_single_item_from_answer_sheets(example_id, answer_sheets_dir, evaluator: Evaluator, vote_pool_dir=None, num_votes=-1):
    data_item = read_json(os.path.join(answer_sheets_dir, f"{example_id}.json"))
    gold_answer = data_item["gold_answer"]

    if num_votes == -1:
        model_answer = data_item["model_answer"]
        if model_answer is None:
            model_answer = evaluator.extract_answer_from_model_completion(data_item["model_completion"])
    elif num_votes > 0:
        assert vote_pool_dir is not None
        votes = read_json(os.path.join(vote_pool_dir, f"{example_id}.json"))
        assert len(votes) > 0
        votes = votes[:num_votes]
        model_answer, _, _, _ = evaluator.find_most_confident_answer(votes)
        
    result = evaluator.check_answers_equiv(model_answer, gold_answer)
    data_item["correctness"] = result

    return data_item


def eval_exp(exp_dir: str, dataset_name: str, num_votes: int = -1):
    answer_sheets_dir = os.path.join(args.exp_dir_path, "answer_sheets")
    vote_pool_dir = os.path.join(args.exp_dir_path, "vote_pool")

    example_ids = [f.replace(".json", "") for f in os.listdir(answer_sheets_dir) if f.endswith(".json")]
    evaluator = eval(f"{dataset_name}Evaluator()")

    with Pool(mp.cpu_count()) as p:
        data_list = list(
            tqdm(
                p.imap_unordered(
                    partial(
                        eval_single_item_from_answer_sheets,
                        answer_sheets_dir=answer_sheets_dir,
                        evaluator=evaluator,
                        vote_pool_dir=vote_pool_dir,
                        num_votes=num_votes,
                    ),
                    example_ids,
                ),
                total=len(example_ids),
            )
        )

    # data_list = [eval_single_item_from_answer_sheets(example_id, answer_sheets_dir, vote_pool_dir, evaluator, num_votes) for example_id in tqdm(example_ids)]

    # Calculate accuracy
    accuracy = sum([item["correctness"] for item in data_list]) / len(data_list)
    print(f"For {answer_sheets_dir}, accuracy: {accuracy}")

    # Save eval results
    eval_result_dir = answer_sheets_dir.replace("run", "eval").replace("answer_sheets", "")
    os.makedirs(eval_result_dir, exist_ok=True)
    save_json(data_list, os.path.join(eval_result_dir, "eval_results.json"))
    analysis = {"accuracy": accuracy}
    save_json(analysis, os.path.join(eval_result_dir, "analysis.json"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--exp_dir_path", type=str, required=True)
    parser.add_argument("--num_votes", type=int, default=-1)
    args = parser.parse_args()

    eval_exp(args.exp_dir_path, args.dataset_name, num_votes=args.num_votes)
