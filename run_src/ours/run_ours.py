import sys

sys.path.append(".")

from common.helpers import fix_seeds, setup_model_parallel, read_json
from common.arguments import get_parser, post_process_args, save_args
from ours_helpers import GeneratorError
from MCTS_for_reasoning import Generator, search_for_answers
from eval_src.Evaluator import GSM8KEvaluator, MATHEvaluator, FOLIOEvaluator, MULTIARITHEvaluator

from tqdm import tqdm
import os, json, time


def main(args):
    fix_seeds(args.seed)
    if args.model_parallel:
        args.local_rank, args.world_size = setup_model_parallel()
    else:
        args.local_rank, args.world_size = 0, 1

    test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist."
    data_item_list = read_json(test_file)

    evaluator = eval(f"{args.dataset_name}Evaluator()")

    tokenizer, model = None, None
    if args.api == "huggingface":
        from models.HuggingFace_API import load_HF_model

        tokenizer, model = load_HF_model(args.model_ckpt)
    elif args.api == "vllm":
        from models.vLLM_API import load_vLLM_model

        tokenizer, model = load_vLLM_model(args.model_ckpt, args.seed, args.tensor_parallel_size, args.half_precision)

    elif args.api == "gpt3.5-turbo":
        from models.OpenAI_API import load_OpenAI_model

        tokenizer, model = load_OpenAI_model(args.model_ckpt)
    generator = Generator(args, tokenizer, model, evaluator)

    total_correct = 0
    total_correct_limit = 0
    num_tested = 0
    start_time = time.time()

    for i, data_item in enumerate(
        (pbar := tqdm(data_item_list, disable=args.local_rank > 0 or args.verbose, position=1))
    ):
        if i < args.start_idx or i >= args.end_idx:
            continue

        problem_id, problem, gt_solution = data_item["id"], data_item["problem"], data_item["solution"]
        gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)

        js = {
            "id": problem_id,
            "problem": problem,
            "model_completion": None,
            "model_answer": None,
            "all_model_completions": {},
            "gold_solution": gt_solution,
            "gold_answer": gt_answer,
        }

        model_solutions, stopping_id, model_all_solutions = [], -1, []

        try:
            model_solutions, stopping_id, model_all_solutions = search_for_answers(
                args=args, user_question=problem, question_id=i, gt_answer=gt_solution, generator=generator
            )
        except GeneratorError as e:
            print(e)
            js["generator_error"] = {
                "source": e.source,
                "io_input": e.io_input,
                "io_output_list": e.io_output_list,
            }
        except Exception as e:
            print(e)
            js["other_error"] = {"text": str(e)}

        num_tested += 1

        if not args.disable_answer_selection:
            assert len(model_solutions) == len(model_all_solutions)
            for rollout_id, (model_solution, model_all_solution) in enumerate(
                zip(model_solutions, model_all_solutions)
            ):
                model_answer = evaluator.extract_answer_from_model_completion(model_solution)
                model_all_answers = [evaluator.extract_answer_from_model_completion(a) for a in model_all_solution]

                correct = evaluator.check_answers_equiv(model_answer, gt_answer)
                correct_limit = any([evaluator.check_answers_equiv(a, gt_answer) for a in model_all_answers])

                if rollout_id == stopping_id:
                    total_correct += int(correct)
                    total_correct_limit += int(correct_limit)
                    js["model_completion"] = model_solution
                    js["model_answer"] = model_answer
                    js["model_all_answer"] = model_all_solution
                js["all_model_completions"][f"rollout_{rollout_id}"] = {
                    "model_solution": model_solution,
                    "model_answer": model_answer,
                    "correct": correct,
                    "correct_limit": correct_limit,
                }

            print(f"accuracy: {total_correct/(num_tested):.3f}")
            print(f"limit accuracy: {total_correct_limit/(num_tested):.3f}")

        with open(os.path.join(args.answer_sheets_dir, f"Question {i:04d} - Answer.json"), "w") as f:
            json.dump(js, f)

        with open(os.path.join(args.run_outputs_dir, "intermediate_result.txt"), "w") as f:
            if not args.disable_answer_selection:
                f.write(f"Num tested: {num_tested}\n")
                f.write(f"Num correct: {total_correct}\n")
                f.write(f"Acc: {total_correct/(num_tested)}\n")
            f.write(
                f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n"
            )
            f.write(
                f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
            )

    end_time = time.time()

    if not args.disable_answer_selection:
        print(f"==> Acc: {total_correct/(num_tested)}")
    print(f"==> Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}")
    print(f"==> Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}")
    print(f"==> Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s")

    with open(os.path.join(args.run_outputs_dir, "final_result.txt"), "w") as f:
        if not args.disable_answer_selection:
            f.write(f"Num tested: {num_tested}\n")
            f.write(f"Num correct: {total_correct}\n")
            f.write(f"Acc: {total_correct/(num_tested)}\n")
        f.write(f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n")
        f.write(
            f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
        )
        f.write(f"Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s\n")


if __name__ == "__main__":
    #! -------------------------------- Arguments --------------------------------
    parser = get_parser()

    parser.add_argument("--num_rollouts", type=int, default=15)
    parser.add_argument(
        "--num_subquestions",
        type=int,
        default=3,
        help="Number of trials for proposing the next subquestion",
    )
    parser.add_argument("--num_votes", type=int, default=10)
    parser.add_argument("--max_depth_allowed", type=int, default=5)
    parser.add_argument("--enable_self_evaluation", action="store_true")

    # MCTS
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--mcts_reward_mode", choices=["path_average", "last_only"], required=True)
    parser.add_argument("--mcts_num_last_votes", type=int, default=None)
    parser.add_argument("--save_tree", action="store_true")

    # Subquestion
    parser.add_argument("--only_enable_subquestion", action="store_true")
    parser.add_argument("--enable_tot_after_subquestion", action="store_true")

    # Paraphrasing
    parser.add_argument("--enable_rephrasing", action="store_true")
    parser.add_argument("--modify_prompts_for_rephrasing", action="store_true")

    # ToT
    parser.add_argument("--enable_tot", action="store_true")
    parser.add_argument("--only_enable_tot", action="store_true")
    parser.add_argument("--num_tot_steps", type=int, default=None)

    # Reasoning cache
    parser.add_argument("--enable_cache", action="store_true")

    #! -------------------------- Used for selecting answer --------------------------
    parser.add_argument("--disable_answer_selection", action="store_true")

    parser.add_argument("--enable_potential_score", action="store_true")

    # Selection Metric
    parser.add_argument(
        "--answer_selection_metric", choices=["select_response", "select_answer"], default="select_response"
    )

    # Selection Mode
    parser.add_argument("--answer_selection_mode", choices=["topk", "adaptive"], default="topk")

    # topk for Selection
    parser.add_argument("--topk", type=int, default=1)
    #! -------------------------------------------------------------------------------

    args = parser.parse_args()

    args.enable_hinted_direct_answer = True

    if args.mcts_reward_mode == "path_average":
        assert args.num_votes > 1
    elif args.mcts_reward_mode == "last_only":
        if args.mcts_num_last_votes is None:
            args.mcts_num_last_votes = 32

    if args.enable_rephrasing:
        assert args.mcts_reward_mode == "last_only", "Rephrasing only supports last_only reward mode"

    if args.enable_tot:
        assert args.mcts_reward_mode == "last_only", "ToT only supports last_only reward mode"
        if args.num_tot_steps is None:
            args.num_tot_steps = 3
        args.enable_hinted_direct_answer = True

    if args.enable_cache:
        assert args.mcts_reward_mode == "path_average"

    #! ----------------------------------------------------------------------------

    prompts_dir = os.path.join(args.prompts_root, args.dataset_name)

    args.fewshot_cot_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
    args.fewshot_cot_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_config.json")

    args.fewshot_tot_prompt_path = os.path.join(prompts_dir, "fewshot_tot", "fewshot_tot_prompt.txt")
    args.fewshot_tot_config_path = os.path.join(prompts_dir, "fewshot_tot", "fewshot_tot_config.json")

    args.decompose_template_path = os.path.join(prompts_dir, "decompose", "decompose_template.json")
    args.decompose_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    if args.enable_self_evaluation:
        args.evaluation_cot_dir = os.path.join(prompts_dir, "evaluation", "cot.txt")
        args.evaluation_decompose_dir = os.path.join(prompts_dir, "evaluation", "decompose.txt")

    if args.enable_rephrasing:
        args.rephrasing_prompt_template_path = os.path.join(prompts_dir, "rephrasing_prompt_template.txt")
        if args.modify_prompts_for_rephrasing:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_cot", "fewshot_cot_prompt_rephrased.txt"
            )
            args.fewshot_tot_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_tot", "fewshot_tot_prompt_rephrased.txt"
            )
            args.decompose_prompt_rephrased_path = os.path.join(
                prompts_dir, "decompose", "decompose_prompt_rephrased.txt"
            )
        else:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
            args.fewshot_tot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_tot", "fewshot_tot_prompt.txt")
            args.decompose_prompt_rephrased_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    args = post_process_args(args)
    print(args)
    save_args(args)
    main(args)
