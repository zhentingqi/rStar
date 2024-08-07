import sys

sys.path.append(".")
from eval_src.Evaluator import GSM8KEvaluator, MATHEvaluator, FOLIOEvaluator, LOGIQAEvaluator
import os, json, re, wandb, time, numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from matplotlib import pyplot as plt
from typing import List, Dict
import multiprocessing as mp
from functools import partial


def select_answer(answer2ratio_sorted: Dict[str, float], answer_selection_stratety: str):
    assert answer_selection_stratety in ["maj", "top_k", "top_p", "mix"]
    if answer_selection_stratety == "maj":
        return max(answer2ratio_sorted, key=answer2ratio_sorted.get)
    elif answer_selection_stratety == "top_k":
        k = 3
        top_k_answers = list(answer2ratio_sorted.keys())[:k]
        # Sample from top k answers with their probabilities (normalized)
        probs = [answer2ratio_sorted[ans] for ans in top_k_answers]
        probs = np.array(probs) / sum(probs)
        return np.random.choice(top_k_answers, p=probs)
    elif answer_selection_stratety == "top_p":
        p = 0.6
        # Cumulative probabilities > p
        top_p_answers = []
        cum_prob = 0
        for ans, prob in answer2ratio_sorted.items():
            cum_prob += prob
            top_p_answers.append(ans)
            if cum_prob > p:
                break
        # Sample from top p answers with their probabilities (normalized)
        probs = [answer2ratio_sorted[ans] for ans in top_p_answers]
        probs = np.array(probs) / sum(probs)
        return np.random.choice(top_p_answers, p=probs)
    elif answer_selection_stratety == "mix":
        if len(answer2ratio_sorted) == 1:
            return list(answer2ratio_sorted.keys())[0]
        else:
            if 2/3 * list(answer2ratio_sorted.values())[0] > list(answer2ratio_sorted.values())[1]:
                return list(answer2ratio_sorted.keys())[0]
            else:
                k = 3
                top_k_answers = list(answer2ratio_sorted.keys())[:k]
                # Sample from top k answers with their probabilities (normalized)
                probs = [answer2ratio_sorted[ans] for ans in top_k_answers]
                probs = np.array(probs) / sum(probs)
                return np.random.choice(top_k_answers, p=probs)


def get_statistics_single_rollout(rollout_id: int, answer_sheets_dir, output_dir, dataset_name: str, plot=False):
    def parse_tree(file_path):
        def get_node_name(input_str: str):
            assert any(name in input_str for name in ["SQ", "DA"])
            # Define the pattern to capture node type and node id
            pattern = r"(SQ|DA)-(\d+):"
            # Search for the pattern in the input string
            match = re.search(pattern, input_str)
            if match:
                # If a match is found, concatenate the node type and id
                node_name = f"{match.group(1)}-{match.group(2)}"
                return node_name
            else:
                # Return None if no match is found
                breakpoint()
                return None

        def get_completion(input_str: str):
            assert any(name in input_str for name in ["SQ-", "DA-"])
            assert "RS-" not in input_str, input_str
            if "A: " in input_str and "SQ-" in input_str:
                return input_str.split("A: ")[-1], input_str.endswith(".")
            elif "Ans: " in input_str and "DA-" in input_str:
                return input_str.split("Ans: ")[-1], input_str.endswith(".")

        def get_score(input_str: str):
            # Define the pattern to capture the score
            pattern = r"V: (\d+(\.\d+)?); UCT: "
            # Search for the pattern in the input string
            match = re.search(pattern, input_str)
            if match:
                # If a match is found, convert the score to float and return
                score = float(match.group(1))
                return score
            else:
                # Return None if no match is found
                return None

        solution_nodes = []
        has_chosen = False
        with open(file_path, "r") as file:
            lines = [l.strip("\n") for l in file.readlines()]
            i = 0
            while i < len(lines):
                if "(T)" in lines[i]:  #! modify this according to your score name (see print_tree())
                    # to determine whether it is a valid solution node
                    assert "----" in lines[i]
                    if "SQ" in lines[i] or "DA" in lines[i]:
                        start, end = i, i + 1
                        while "====" not in lines[end] and "----" not in lines[end]:
                            end += 1
                        section = "\n".join(lines[start:end])
                        increment = end - start
                    else:
                        print(file_path)
                        print(lines[i])
                        raise ValueError("Terminal node should be either SQ or DA")

                    node_name = get_node_name(section)
                    model_completion, ends_properly = get_completion(section)
                    score = get_score(section)
                    if "[[" in lines[i] and "]]" in lines[i]:
                        assert has_chosen == False
                        has_chosen = True
                        solution_nodes.append(
                            {
                                "node_name": node_name,
                                "model_completion": model_completion,
                                "model_answer": str(evaluator.extract_answer_from_model_completion(model_completion)),
                                "score": score,
                                "chosen": True,
                                "ends_properly": ends_properly,
                            }
                        )
                        chosen_solution_node = solution_nodes[-1]
                    else:
                        solution_nodes.append(
                            {
                                "node_name": node_name,
                                "model_completion": model_completion,
                                "model_answer": str(evaluator.extract_answer_from_model_completion(model_completion)),
                                "score": score,
                                "chosen": False,
                                "ends_properly": ends_properly,
                            }
                        )
                else:
                    increment = 1
                i += increment

        assert has_chosen, file_path
        return solution_nodes, chosen_solution_node

    def frac(a, b):
        return a / b if b != 0 else 0

    evaluator = eval(f"{dataset_name}Evaluator()")

    # ----------------------------- Collect total files count -----------------------------
    total_files_cnt = 0
    recorded_correct_cnt = 0
    for filename in os.listdir(answer_sheets_dir):
        if filename.startswith("Question") and filename.endswith("Answer.json"):
            total_files_cnt += 1
            file_path = os.path.join(answer_sheets_dir, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                try:
                    recorded_correct_cnt += int(data["all_model_completions"][f"rollout_{rollout_id}"]["correct"])
                except:
                    print(f"Error in file {file_path}\n")
                    assert any("error" in key for key in data.keys())
                    total_files_cnt -= 1
                    pass

    assert total_files_cnt > 0
    # -------------------------------------------------------------------------------------

    correct_cnt = 0
    any_correct_cnt = 0
    correct_cnt_scorebased = 0
    correct_cnt_maj = 0
    correct_cnt_top_k = 0
    correct_cnt_top_p = 0
    correct_cnt_mix = 0
    V_of_chosen_solution_node_list = []
    ratio_of_correct_solution_node_list = []
    num_solution_node_list = []
    DA_pos_SQ_pos, DA_pos_SQ_neg, DA_neg_SQ_pos, DA_neg_SQ_neg = 0, 0, 0, 0
    DA_chosen_cnt, SQ_chosen_cnt = 0, 0
    DA_chosen_correct_cnt, SQ_chosen_correct_cnt = 0, 0
    not_ends_properly_rate_sum = 0

    for filename in os.listdir(answer_sheets_dir):
        if filename.startswith("Question") and filename.endswith(f"Rollout {rollout_id}.tree"):
            #! Get ground truth answer
            answer_file_path = os.path.join(
                answer_sheets_dir, filename.replace(f"Rollout {rollout_id}.tree", "Answer.json")
            )
            if not os.path.exists(answer_file_path):
                continue
            with open(answer_file_path, "r") as f:
                js = json.load(f)
                gt_answer = js["gold_answer"]

            #! Parse tree
            tree_file_path = os.path.join(answer_sheets_dir, filename)
            solution_nodes, chosen_solution_node = parse_tree(tree_file_path)

            if evaluator.check_answers_equiv(gt_answer, chosen_solution_node["model_answer"]):
                correct_cnt += 1

            answer2cnt = {}
            for st in solution_nodes:
                st_answer = st["model_answer"]
                has_existed = False
                for existing_answer in answer2cnt.keys():
                    if evaluator.check_answers_equiv(existing_answer, st_answer):
                        assert not has_existed
                        has_existed = True
                        answer2cnt[existing_answer] += 1
                if not has_existed:
                    answer2cnt[st_answer] = 1
            
            answer2ratio = {k: frac(v, len(solution_nodes)) for k, v in answer2cnt.items()}
            answer2ratio = dict(sorted(answer2ratio.items(), key=lambda x: x[1], reverse=True))

            #! Try different answer selection strategies
            selected_answer_scorebased = max([st for st in solution_nodes], key=lambda x: x["score"])["model_answer"]
            correct_cnt_scorebased += int(evaluator.check_answers_equiv(gt_answer, selected_answer_scorebased))
            selected_answer_maj = select_answer(answer2ratio, "maj")
            correct_cnt_maj += int(evaluator.check_answers_equiv(gt_answer, selected_answer_maj))
            selected_answer_top_k = select_answer(answer2ratio, "top_k")
            correct_cnt_top_k += int(evaluator.check_answers_equiv(gt_answer, selected_answer_top_k))
            selected_answer_top_p = select_answer(answer2ratio, "top_p")
            correct_cnt_top_p += int(evaluator.check_answers_equiv(gt_answer, selected_answer_top_p))
            selected_answer_mix = select_answer(answer2ratio, "mix")
            correct_cnt_mix += int(evaluator.check_answers_equiv(gt_answer, selected_answer_mix))

            #! Draw distributions of answers v.s. ratio
            if plot:
                if evaluator.check_answers_equiv(gt_answer, chosen_solution_node["model_answer"]):
                    plot_dir = os.path.join(output_dir, f"rollout_{rollout_id}", "correct_examples")
                else:
                    plot_dir = os.path.join(output_dir, f"rollout_{rollout_id}", "incorrect_examples")

                os.makedirs(plot_dir, exist_ok=True)
                target_file = os.path.join(plot_dir, f"{filename.split('-')[0].strip()}.png")
                plt.figure(figsize=(10, 5))
                colors = ["red" if evaluator.check_answers_equiv(gt_answer, ans) else "blue" for ans in answer2ratio.keys()]
                
                # Check only one red
                has_red = False
                for c in colors:
                    if c == "red":
                        assert not has_red, print(f"Multiple reds in {filename} at rollout {rollout_id}: {answer2ratio}")
                        has_red = True
                        
                plt.bar(answer2ratio.keys(), answer2ratio.values(), color=colors)
                plt.xlabel("Answer")
                plt.ylabel("Ratio")
                plt.title(f"Answer distribution of {filename}")
                plt.savefig(target_file)
                plt.close()
                
            #! Collect statistics
            find_correct = False
            num_correct_solution_node = 0
            for st in solution_nodes:
                if evaluator.check_answers_equiv(gt_answer, st["model_answer"]):
                    num_correct_solution_node += 1
                    find_correct = True
            if find_correct:
                any_correct_cnt += 1

            V_of_chosen_solution_node_list.append(chosen_solution_node["score"])
            ratio_of_correct_solution_node_list.append(frac(num_correct_solution_node, len(solution_nodes)))
            num_solution_node_list.append(len(solution_nodes))

            DA_correct = False
            SQ_correct = False
            for st in solution_nodes:
                if "DA" in st["node_name"]:
                    correct = evaluator.check_answers_equiv(gt_answer, st["model_answer"])
                    if correct:
                        DA_correct = True
                    chosen = st["chosen"]
                    DA_chosen_cnt += int(chosen)
                    DA_chosen_correct_cnt += int(correct and chosen)
                elif "SQ" in st["node_name"]:
                    correct = evaluator.check_answers_equiv(gt_answer, st["model_answer"])
                    if correct:
                        SQ_correct = True
                    chosen = st["chosen"]
                    SQ_chosen_cnt += int(chosen)
                    SQ_chosen_correct_cnt += int(correct and chosen)
                else:
                    raise ValueError

            if DA_correct and SQ_correct:
                DA_pos_SQ_pos += 1
            elif DA_correct and not SQ_correct:
                DA_pos_SQ_neg += 1
            elif not DA_correct and SQ_correct:
                DA_neg_SQ_pos += 1
            elif not DA_correct and not SQ_correct:
                DA_neg_SQ_neg += 1
            else:
                raise ValueError

            not_ends_properly_rate = len([n for n in solution_nodes if not n["ends_properly"]]) / len(solution_nodes)
            not_ends_properly_rate_sum += not_ends_properly_rate

    assert DA_pos_SQ_pos + DA_pos_SQ_neg + DA_neg_SQ_pos + DA_neg_SQ_neg == total_files_cnt
    assert any_correct_cnt == DA_pos_SQ_pos + DA_pos_SQ_neg + DA_neg_SQ_pos
    assert any_correct_cnt >= correct_cnt

    return {
        "rollout id": rollout_id,
        "Important / acc": frac(correct_cnt, total_files_cnt),
        "Important / acc (scorebased)": frac(correct_cnt_scorebased, total_files_cnt),
        "Important / acc (maj)": frac(correct_cnt_maj, total_files_cnt),
        "Important / acc (top_k)": frac(correct_cnt_top_k, total_files_cnt),
        "Important / acc (top_p)": frac(correct_cnt_top_p, total_files_cnt),
        "Important / acc (mix)": frac(correct_cnt_mix, total_files_cnt),
        "Important / limit acc": frac(any_correct_cnt, total_files_cnt),
        "Important / gap": frac(any_correct_cnt - correct_cnt, total_files_cnt),
        "Important / avg. V of chosen solution node": sum(V_of_chosen_solution_node_list) / total_files_cnt,
        "Important / avg. ratio of correct solution nodes": sum(ratio_of_correct_solution_node_list) / total_files_cnt,
        "Important / avg. num of solution nodes": sum(num_solution_node_list) / total_files_cnt,
        "DA&SQ / DA correct rate": frac(DA_pos_SQ_pos + DA_pos_SQ_neg, total_files_cnt),
        "DA&SQ / SQ correct rate": frac(DA_pos_SQ_pos + DA_neg_SQ_pos, total_files_cnt),
        "DA&SQ / DA (+) SQ (+) rate": frac(DA_pos_SQ_pos, total_files_cnt),
        "DA&SQ / DA (+) SQ (-) rate": frac(DA_pos_SQ_neg, total_files_cnt),
        "DA&SQ / DA (-) SQ (+) rate": frac(DA_neg_SQ_pos, total_files_cnt),
        "DA&SQ / DA (-) SQ (-) rate": frac(DA_neg_SQ_neg, total_files_cnt),
        "DA&SQ / DA chosen rate": frac(DA_chosen_cnt, total_files_cnt),
        "DA&SQ / DA chosen correct over DA chosen": frac(DA_chosen_correct_cnt, DA_chosen_cnt),
        "DA&SQ / DA chosen correct over DA correct": frac(DA_chosen_correct_cnt, DA_pos_SQ_pos + DA_pos_SQ_neg),
        "DA&SQ / SQ chosen rate": frac(SQ_chosen_cnt, total_files_cnt),
        "DA&SQ / SQ chosen correct over SQ chosen": frac(SQ_chosen_correct_cnt, SQ_chosen_cnt),
        "DA&SQ / SQ chosen correct over SQ correct": frac(SQ_chosen_correct_cnt, DA_pos_SQ_pos + DA_neg_SQ_pos),
        "Others / not ends properly avg rate": frac(not_ends_properly_rate_sum, total_files_cnt),
    }


if __name__ == "__main__":
    answer_sheets_dir_list = [
    ]
    for answer_sheets_dir in answer_sheets_dir_list:
        assert os.path.exists(answer_sheets_dir)

        allowed_names = ["GSM8K", "FOLIO", "MATH"]
        for name in allowed_names:
            if name in answer_sheets_dir:
                dataset_name = name
                break

        output_dir = f"out/analysis/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        num_rollouts = 64
            
        print(f"Processing {answer_sheets_dir} for rollout {num_rollouts}...")
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        partial(
                            get_statistics_single_rollout,
                            answer_sheets_dir=answer_sheets_dir,
                            output_dir=output_dir,
                            dataset_name=dataset_name,
                        ),
                        range(num_rollouts),
                    ),
                    total=num_rollouts,
                )
            )
        for statistics in results:
            rollout_id = list(statistics.values())[0]["rollout id"]
            has_correct_solution_node = list(statistics.values())[0]["Important / limit acc"] > 0
            rollout_id = statistics["rollout id"]
            has_correct_solution_node = statistics["Important / limit acc"] > 0
            
        print("Sleeping for 5 seconds...")
        time.sleep(5)
        