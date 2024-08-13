import sys, re

sys.path.append(".")

from common.helpers import fix_seeds, read_json, read_txt
from eval_src.Evaluator import *
from run_src.rstar_helpers import concat_solution_trace, concat_subq_suba_trace, concat_rap_solution_trace, mask_solution_trace, mask_subq_suba_trace, Node_Type
from models.vLLM_API import load_vLLM_model, generate_with_vLLM_model
from models.OpenAI_API import generate_with_OpenAI_model

from tqdm import tqdm
import os, json, time, wandb, random, gc, torch
from argparse import ArgumentParser
from itertools import combinations
from collections import defaultdict
from copy import deepcopy
from time import time
from datetime import datetime
import multiprocessing as mp
from functools import partial
from copy import deepcopy


class Candidate:
    def __init__(self, solution_trace, masked_solution_trace_list, final_step, final_answer, id, freq=1, trace_reward=1., c_type="default"):
        self.solution_trace = solution_trace
        self.masked_solution_trace_list = masked_solution_trace_list
        self.final_step = final_step
        self.final_answer = final_answer
        self.id = id
        self.freq = freq
        self.trace_reward = trace_reward
        self.c_type = c_type

    def __str__(self):
        return f"Candidate {self.id}: {self.final_answer}"

    def to_dict(self):
        return {
            "solution_trace": self.solution_trace,
            "masked_solution_trace_list": self.masked_solution_trace_list,
            "final_step": self.final_step,
            "final_answer": self.final_answer,
            "id": self.id
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            solution_trace=data["solution_trace"],
            masked_solution_trace_list=data["masked_solution_trace_list"],
            final_step=data["final_step"],
            final_answer=data["final_answer"],
            id=data["id"]
        )


def group_completions_by_answer(completions: list[str], evaluator):
    answer2completions = {}
    for c in completions:
        has_existed = False
        for existing_answer in answer2completions.keys():
            if evaluator.check_answers_equiv(c, existing_answer):
                has_existed = True
                answer2completions[existing_answer].append(c)
                break

        if not has_existed:
            answer2completions[evaluator.extract_answer_from_model_completion(c)] = [c]

    answer2confidence = {ans: len(comps) / len(completions) for ans, comps in answer2completions.items()}
    answer2cnt = {ans: len(comps) for ans, comps in answer2completions.items()}

    return answer2completions, answer2confidence, answer2cnt


def group_candidates_by_answer(candidates: list[Candidate], evaluator, criteria="freq"):
    """Return answer2candidates, answer2confidence, answer2cnt."""
    answer2candidates = {}
    answer2confidence = defaultdict(float)
    answer2cnt = defaultdict(int)

    for c in candidates:
        has_existed = False
        for existing_answer in answer2candidates.keys():
            if evaluator.check_answers_equiv(c.final_answer, existing_answer):
                has_existed = True
                answer2candidates[str(existing_answer)].extend([c] * c.freq)
                answer2confidence[str(existing_answer)] += c.trace_reward if criteria == "reward" else c.freq
                answer2cnt[str(existing_answer)] += c.freq
                break

        if not has_existed:
            if str(c.final_answer) in answer2candidates:
                answer2candidates[str(c.final_answer)].extend([c] * c.freq)
            else:
                answer2candidates[str(c.final_answer)] = [c] * c.freq
            answer2confidence[str(c.final_answer)] += c.trace_reward if criteria == "reward" else c.freq
            answer2cnt[str(c.final_answer)] += c.freq

    assert all(answer2cnt[ans] == len(answer2candidates[ans]) for ans in answer2cnt.keys())
    assert float(sum([candidate.trace_reward for candidate in candidates])) == float(sum([answer2confidence[ans] for ans in answer2confidence.keys()]))

    candidates_count = sum([candidate.freq for candidate in candidates])
    for ans in answer2confidence.keys():
        answer2confidence[ans] /= candidates_count

    return answer2candidates, answer2confidence, answer2cnt


class Discriminator:
    def __init__(self, args, evaluator):
        self.args = args
        self.evaluator = evaluator

        self.fewshot_config = read_json(args.fewshot_config_path)
        self.fewshot_template = self.fewshot_config["prompt_template"]
        self.stop_tokens = self.fewshot_config["stop_tokens"]

        self.fewshot_prompt = read_txt(args.fewshot_prompt_path)
        self.fewshot_prompt2 = read_txt("./prompts/GSM8K/fewshot_cot/fewshot_cot_prompt_parse_subq.txt")

    def _filter_none(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if c.final_answer is not None]
        return candidates

    def _filter_long(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if len(c.final_answer) <= 100]
        return candidates

    def _filter_reasoning_consistency(self, gen_model, problem: str, candidates: list[Candidate], aux={}) -> list[Candidate]:
        problem_id = aux["problem_id"]
        file_idx = aux["file_idx"]

        prompt_template = self.fewshot_template
        fewshot_examples = self.fewshot_prompt
        fewshot_examples2 = self.fewshot_prompt2
        stop_tokens = self.stop_tokens

        assert all(len(c.masked_solution_trace_list) == self.args.num_masked_solution_traces for c in candidates if c.c_type == "default")
        gen_input_list = []
        ground_truth_list = []
        c_completion_num_list = []
        for c in candidates:
            if c.c_type == "default":
                for masked_solution_trace in c.masked_solution_trace_list:
                    for _ in range(self.args.rc_n_completions):
                        gen_input_list.append(
                            prompt_template.format(examples=fewshot_examples, instruction=problem) + masked_solution_trace
                        )
                        ground_truth_list.append(c.final_answer)
            else:
                for masked_solution_trace in c.masked_solution_trace_list:
                    if masked_solution_trace[0].endswith("Now we can answer the question:"):
                        match = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$', problem)
                        overall_question = match[1].strip() if match else problem.strip()
                        masked_solution_trace[0] = masked_solution_trace[0].strip() + " " + overall_question[0].upper() + overall_question[1:] + "\nSubanswer: "
                    for _ in range(self.args.rc_n_completions):
                        gen_input_list.append(
                            prompt_template.format(examples=fewshot_examples2, instruction=problem) + masked_solution_trace[0] # TODO: change prompt
                        )
                        ground_truth_list.append(masked_solution_trace[1])
            c_completion_num_list.append(len(c.masked_solution_trace_list)*self.args.rc_n_completions)
        """gen_input_list:
        [c1_mask1, c1_mask2, ..., c2_mask1, c2_mask2, ..., ......, ct_mask1, ct_mask2, ...]
        """

        # Manually split into batches
        batch_size = self.args.max_num_seqs // self.args.rc_n_completions // 2
        gen_output_list = []
        for start_idx in range(0, len(gen_input_list), batch_size):
            end_idx = start_idx + batch_size
            sub_gen_input_list = gen_input_list[start_idx:end_idx]
            sub_gen_output_list = self._gen_func(gen_model=gen_model, gen_input=sub_gen_input_list, temperature=self.args.rc_temperature, n=1, max_tokens=512, stop_tokens=stop_tokens + ["\n"])
            gen_output_list.extend(sub_gen_output_list)

        with open(os.path.join(self.args.discriminate_results_dir, f"problem-{problem_id}.json"), "w") as f:
            js = {
                "problem_id": problem_id,
                "file_idx": file_idx,
                "gen_output_list": gen_output_list
            }
            json.dump(js, f)

        """gen_output_list:
        [[c1_mask1_o1, c1_mask1_o2, ...], [c1_mask2_o1, c1_mask2_o2, ...], ..., [ct_mask1_o1, ct_mask1_o2, ...], [ct_mask2_o1, ct_mask2_o2, ...], ...]
        """

        if all(isinstance(item, list) for item in gen_output_list):
            completion_list = []
            for n_completions in gen_output_list:
                for completion in n_completions:
                    completion_list.append(completion)
            assert len(completion_list) == self.args.rc_n_completions * self.args.num_masked_solution_traces * len(candidates)
            candidate_group_size = self.args.rc_n_completions * self.args.num_masked_solution_traces
        elif all(isinstance(item, str) for item in gen_output_list):
            completion_list = gen_output_list
            candidate_group_size = self.args.num_masked_solution_traces

        answer_list = [
            self.evaluator.extract_answer_from_model_completion(completion) for completion in completion_list
        ]

        count = 0
        completion_group_list = []
        answer_group_list = []
        gt_group_list = []
        for num in c_completion_num_list:
            completion_group_list.append(completion_list[count : count + num])
            answer_group_list.append(answer_list[count : count + num])
            gt_group_list.append(ground_truth_list[count : count + num])
            count += num
        assert count == len(completion_list) == len(answer_list)

        consistent_candidates = []

        for c, completion_group, answer_group, gt_answer in zip(candidates, completion_group_list, answer_group_list, gt_group_list):
            candidate_group_size = len(c.masked_solution_trace_list)
            num_consistent = 0
            if self.args.rc_mode == 'maj':
                answer = self.evaluator.find_most_confident_answer(completion_group)[0]
                if self.evaluator.check_answers_equiv(gt_answer[-1], answer):  # consistent
                    consistent_candidates.append(c)
            else:
                for answer, gt_a in zip(answer_group, gt_answer):
                    if self.evaluator.check_answers_equiv(gt_a, answer):  # consistent
                        num_consistent += 1
                if self.args.rc_mode == 'loose':
                    if num_consistent > 0:
                        consistent_candidates.append(c)
                elif self.args.rc_mode == 'mid':
                    if num_consistent >= candidate_group_size // 2:
                        consistent_candidates.append(c)
                elif self.args.rc_mode == 'strict':
                    if num_consistent == candidate_group_size:
                        consistent_candidates.append(c)

        return consistent_candidates

    def _make_multichoice(self, problem: str, cand1: Candidate, cand2: Candidate):
        prompt = f"Problem: {problem}\n\n"
        prompt += "Which of the following is a solution to the problem above?\n\n"
        prompt += f"A) {cand1.solution_trace}\n"
        prompt += f"B) {cand2.solution_trace}\n"
        prompt += "C) None of the above.\n"
        prompt += "\n"
        prompt += (
            'Please choose the best option. IMPORTANT: End your answer with "The correct option is <A or B or C>".\n\n'
        )
        return prompt

    def _make_multichoice_v2(self, problem: str, cand1: Candidate, cand2: Candidate):
        prompt = f"Problem: {problem}\n\n"
        prompt += "Here are two possible solutions to the problem:\n"
        prompt += f"A) {cand1.solution_trace}\n"
        prompt += f"B) {cand2.solution_trace}\n"
        prompt += "\n"
        prompt += "It could be that one of the solutions is correct or both are incorrect, but you should choose a better one according to their reasoning. "
        prompt += 'Which solution do you think is better? Format your answer as "A better solution is <A or B>".\n\n'
        return prompt

    def _make_multichoice_v3(self, problem: str, cand1: Candidate, cand2: Candidate):
        prompt = f"Problem: {problem}\n\n"
        prompt += "Here are two possible solutions to the problem:\n"
        prompt += f"A) {cand1.solution_trace}\n"
        prompt += f"B) {cand2.solution_trace}\n"
        prompt += "\n"
        prompt += "Either one of the solutions is correct, or both of them are incorrect. In any case, you should choose a better one by checking their reasoning process. "
        prompt += 'Which solution do you think is better? Analyze two solutions step by step, and end your answer as "Solution <A or B> is better".\n\n'
        prompt += "Your response: "
        prompt += "Let's think step by step."
        return prompt

    def _make_multichoice_v4(self, problem: str, cand1: Candidate, cand2: Candidate, prompt_type: str):
        multi_choice_context = "A problem is given below:\n"
        multi_choice_context += f"{problem}\n\n"
        multi_choice_context += "Two potential solutions are provided below:\n"
        multi_choice_context += f"- Solution 1. {cand1.solution_trace}\n"
        multi_choice_context += f"- Solution 2. {cand2.solution_trace}\n"
        multi_choice_context += "\n"
        multi_choice_context += "Choose the best description of the two solutions from the following options:\n"
        multi_choice_context += "A) Solution 1 is correct.\n"
        multi_choice_context += "B) Solution 2 is correct.\n"
        multi_choice_context += "C) Both solutions are correct.\n"
        multi_choice_context += "D) Neither solution is correct.\n"

        if prompt_type == "fewshot":
            demonstrations = """### Instruction:
A problem is given below:
Lee rears only sheep and geese on his farm.  If the total number of animal legs is 70, and the total number of animal heads is 20, how many sheep live on Lee's farm?

Two potential solutions are provided below:
- Solution 1. Let's think step by step. Each goose has 2 legs and one head. If the total number of animal legs is 70, then the number of goats is 70 / 2 = 35. Now we can answer the question: Each sheep has 4 legs and one head. There are 20 heads, so the number of sheep is 20 / 4 = 5. The answer is 5.
- Solution 2. Let's think step by step. There are 20 animal heads in total. Geese have two heads while sheep do not. This implies that all 20 heads belong to 20/2 = 10 geese. Now we can answer the question: There are 70 animal legs in total. Sheep have four legs but geese don't. This implies that 70 - 10*2 = 50 legs belong to 50/4 = 12 sheep. The answer is 12.

Choose the best description of the two solutions from the following options:
A) Solution 1 is correct.
B) Solution 2 is correct.
C) Both solutions are correct.
D) Neither solution is correct.

### Response:
Solution 1 is incorrect because it mistakenly divides the number of legs by 2 to get the number of geese, and then incorrectly uses the number of heads to calculate the number of sheep. Solution 2 is also incorrect because it incorrectly assumes geese have two heads and makes errors in the calculations of the number of geese and sheep. Therefore, both solutions contain significant logical and mathematical errors. The correct option is D.

### Instruction:
A problem is given below:
Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant. Based on the above information, is the following statement true, false, or uncertain? A Czech published a book in 1946.

Two potential solutions are provided below:
- Solution 1. Let's think step by step. Miroslav Venhoda was a Czech choral conductor who published a book in 1946 called Method of Studying Gregorian Chant. Since Miroslav Venhoda is Czech and he published a book in 1946, the statement that "A Czech published a book in 1946" is supported by the information provided. The answer is: True.
- Solution 2. Miroslav Venhoda was a Czech choral conductor who specialized in Renaissance and Baroque music. Some musicians love music, and any choral conductor is a musician. However, nothing in the given information directly states that a Czech published a book in 1946. The answer is: Uncertain.

Choose the best description of the two solutions from the following options:
A) Solution 1 is correct.
B) Solution 2 is correct.
C) Both solutions are correct.
D) Neither solution is correct.

### Response:
Solution 1 clearly identifies that Miroslav Venhoda, a Czech choral conductor, published a book in 1946 called Method of Studying Gregorian Chant. Since Venhoda is Czech and published a book in 1946, the statement "A Czech published a book in 1946" is true based on the provided facts. Solution 2, however, focuses on the general statement about musicians and their love for music and fails to acknowledge the direct evidence that Miroslav Venhoda, a Czech, published a book in 1946. By incorrectly stating that the information is uncertain, it overlooks the specific detail given about Venhoda's nationality and his publication. The correct option is A."""

            prompt = 'A chat between a curious user and an AI assistant. At the end of assistant\'s response, a final answer is given in the format of "The correct option is <A or B or C or D>".\n'
            prompt += "\n"
            prompt += demonstrations
            prompt += "\n"
            prompt += "\n"
            prompt += "### Instruction:\n"
            prompt += multi_choice_context
            prompt += "\n"
            prompt += "### Response:\n"
            return prompt
        elif prompt_type == "instruct":
            prompt = multi_choice_context
            prompt += "\n"
            prompt += 'Analyze the two solutions, and choose an option from A, B, C, and D. IMPORTANT: At the end of your response, a final answer should be given in the format of "The correct option is <A or B or C or D>".\n'
            prompt += "\n"
            prompt += "Your response:\n"
            return prompt

    def _parse_multichoice_v4(self, gen_output: str) -> str:
        gen_output = (
            gen_output.replace(":", "")
            .replace("\n", "")
            .replace("would be", "is")
            .replace(" therefore ", " ")
            .replace(" thus ", " ")
            .replace(" a ", " ")
            .replace(" the ", " ")
        )
        if any(
            x in gen_output
            for x in [
                "correct option A",
                "option is A",
                "option is <A>",
                "choice is A",
                "choice is option A",
                "choice is <A>",
                "A is better",
                "A is best",
                "A seems to be better",
                "A is more",
                "A is most",
                "A is correct",
                "answer is A",
                "Solution 1 is correct",
                "Solution 2 is incorrect",
            ]
        ):
            return "A"
        elif any(
            x in gen_output
            for x in [
                "correct option B",
                "option is B",
                "option is <B>",
                "choice is B",
                "choice is option B",
                "choice is <B>",
                "B is better",
                "B is best",
                "B seems to be better",
                "B is more",
                "B is most",
                "B is correct",
                "answer is B",
                "Solution 2 is correct",
                "Solution 1 is incorrect",
            ]
        ):
            return "B"
        elif any(
            x in gen_output
            for x in [
                "correct option C",
                "option is C",
                "option is <C>",
                "choice is C",
                "choice is option C",
                "choice is <C>",
                "C is better",
                "C is best",
                "C seems to be better",
                "C is more",
                "C is most",
                "C is correct",
                "answer is C",
                "Both",
                "both",
            ]
        ):
            return "C"
        elif any(
            x in gen_output
            for x in [
                "correct option D",
                "option is D",
                "option is <D>",
                "choice is D",
                "choice is option D",
                "choice is <D>",
                "D is better",
                "D is best",
                "D seems to be better",
                "D is more",
                "D is most",
                "D is correct",
                "answer is D",
                "neither",
                "Neither",
                "none of",
                "None of",
            ]
        ):
            return "D"
        else:
            return None

    def _gen_func(self, gen_model, gen_input, temperature: float, n: int=1, max_tokens: int = 768, stop_tokens=None):
        if temperature == 0.0:
            n = 1

        if self.args.api == "vllm":
            response = generate_with_vLLM_model(
                model=gen_model, input=gen_input, temperature=temperature, n=n, max_tokens=max_tokens, stop=stop_tokens
            )
            if n == 1:
                if isinstance(gen_input, str):
                    return response[0].outputs[0].text
                elif isinstance(gen_input, list):
                    return [r.outputs[0].text for r in response]
            elif n > 1:
                if isinstance(gen_input, str):
                    return [o.text for o in response[0].outputs]
                elif isinstance(gen_input, list):
                    return [[o.text for o in r.outputs] for r in response]
        elif self.args.api == "gpt-3.5-turbo":
            response = []
            if isinstance(gen_input, str):
                gen_input = [gen_input]
            for p in tqdm(gen_input):
                response.append(generate_with_OpenAI_model(
                    prompt=p, temperature=temperature, max_tokens=max_tokens, stop=stop_tokens
                ))
            if n == 1:
                if isinstance(gen_input, str):
                    return response[0]
                elif isinstance(gen_input, list):
                    return response
            elif n > 1:
                raise NotImplementedError
                import pdb; pdb.set_trace()
                if isinstance(gen_input, str):
                    return [o.text for o in response[0].outputs]
                elif isinstance(gen_input, list):
                    return [[o.text for o in r.outputs] for r in response]

        elif self.args.api == "together":
            raise NotImplementedError
            if isinstance(gen_input, str):
                return text_completion(
                    prompt=gen_input, model_ckpt=self.args.model_ckpt, max_tokens=max_tokens, temperature=temperature, stop=stop_tokens
                )
            elif isinstance(gen_input, list):
                return text_completion_concurrent(
                    prompt_list=gen_input, model_ckpt=self.args.model_ckpt, max_tokens=max_tokens, temperature=temperature, stop=stop_tokens
                )

    def _calculate_scores(self, unfiltered_candidates: list[Candidate], filtered_candidates: list[Candidate]) -> dict:
        _, filtered_answer2confidence, filtered_answer2cnt = group_candidates_by_answer(filtered_candidates, self.evaluator, self.args.rc_criteria)
        print(f"==> Confidence: {filtered_answer2confidence}")
        _, _, unfiltered_answer2cnt = group_candidates_by_answer(unfiltered_candidates, self.evaluator, self.args.rc_criteria)

        filtered_answer2survival_rate = {}
        for filtered_ans in filtered_answer2cnt.keys():
            has_existed = False
            for unfiltered_ans in unfiltered_answer2cnt.keys():
                if self.evaluator.check_answers_equiv(filtered_ans, unfiltered_ans):
                    has_existed = True
                    filtered_answer2survival_rate[filtered_ans] = filtered_answer2cnt[filtered_ans] / unfiltered_answer2cnt[unfiltered_ans]
                    break
            if not has_existed:
                filtered_answer2survival_rate[filtered_ans] = 0.0

        print(f"==> Survival rates: {filtered_answer2survival_rate}")

        filtered_answer2score = {}
        for filtered_ans in filtered_answer2confidence.keys():
            has_existed = False
            for unfiltered_ans in unfiltered_answer2cnt.keys():
                if self.evaluator.check_answers_equiv(filtered_ans, unfiltered_ans):
                    has_existed = True
                    filtered_answer2score[filtered_ans] = filtered_answer2confidence[filtered_ans] + filtered_answer2survival_rate[filtered_ans]
                    break
            if not has_existed:
                filtered_answer2score[filtered_ans] = 0.0

        print(f"==> Scores: {filtered_answer2score}")

        return filtered_answer2score

    def _find_winner_naive(self, candidates: list[Candidate], gt_answer: str = None) -> Candidate:
        print(f"==> Find most confident from: {[c.final_answer for c in candidates]}")

        if len(candidates) == 0:
            winner = None
            print(f"==> Winner answer: None")
        elif len(candidates) == 1:
            winner = candidates[0]
            print(f"==> Winner answer: {winner.final_answer}\n")
        elif not any(self.evaluator.check_answers_equiv(c.final_answer, gt_answer) for c in candidates):
            winner = None
            print(f"==> Winner answer: None")
        else:
            answer2candidates, answer2confidence, _ = group_candidates_by_answer(candidates, self.evaluator, self.args.rc_criteria)
            print(f"==> Confidence: {answer2confidence}")
            most_confident_answer = max(answer2confidence.keys(), key=lambda x: answer2confidence[x])
            winner = answer2candidates[most_confident_answer][0]
            print(f"==> Winner answer: {most_confident_answer}\n")

        return winner

    def _find_winner_filtered(self, unfiltered_candidates: list[Candidate], filtered_candidates: list[Candidate], gt_answer: str = None) -> Candidate:
        if len(filtered_candidates) == 0:
            answer2candidates, answer2confidence, _ = group_candidates_by_answer(unfiltered_candidates, self.evaluator, self.args.rc_criteria)
            most_confident_answer = max(answer2confidence.keys(), key=lambda x: answer2confidence[x])
            winner = answer2candidates[most_confident_answer][0]
            print(f"==> Winner answer: {most_confident_answer}\n")
        elif len(filtered_candidates) == 1:
            winner = filtered_candidates[0]
            print(f"==> Winner answer: {winner.final_answer}\n")
        elif not any(self.evaluator.check_answers_equiv(c.final_answer, gt_answer) for c in filtered_candidates):
            winner = None
            print(f"==> Winner answer: None")
        else:
            filtered_answer2score = self._calculate_scores(unfiltered_candidates, filtered_candidates)
            winner_answer = max(filtered_answer2score.keys(), key=lambda x: filtered_answer2score[x])
            print(f"==> Winner answer: {winner_answer}")
            winner = next(
                c for c in filtered_candidates if self.evaluator.check_answers_equiv(c.final_answer, winner_answer)
            )

        return winner

    def select(self, problem: str, candidates: list[Candidate], gt_answer: str = None, aux={}) -> Candidate:
        raise NotImplementedError()


class MajorityVoteDiscriminator(Discriminator):
    def __init__(self, args, evaluator):
        super().__init__(args, evaluator)
        self.tokenizer, self.model = None, None
        if self.args.api == "vllm":
            self.tokenizer, self.model = load_vLLM_model(args.model_ckpt, args.seed, max_num_seqs=args.max_num_seqs)

    def select(self, problem: str, candidates: list[Candidate], gt_answer: str = None, aux={}) -> Candidate:
        print(f"==> Ground truth answer: {gt_answer}")

        unfiltered_candidates = candidates
        print(f"==> Unfiltered answers: {[c.final_answer for c in unfiltered_candidates]}")
        # candidate: [1, 2, 3, 4, 5, None, paosdifjpsod]
        prefiltered_candidates = self._filter_none(candidates)
        prefiltered_candidates = self._filter_long(prefiltered_candidates)
        # prefiltered_candidates: [1, 2, 3, 4, 5]
        print(f"==> Pre-filtered answers: {[c.final_answer for c in prefiltered_candidates]}")
        filtered_candidates = self._filter_reasoning_consistency(self.model, problem, prefiltered_candidates, aux)
        # filtered_candidates: [1, 2, 3]
        print(f"==> RC-filtered answers: {[c.final_answer for c in filtered_candidates]}")

        if self.args.save_inconsistent_candidates:
            # vote: [1, 2, 3] + [1, 2, 3, 4, 5] (enhance)
            return self._find_winner_naive(candidates=prefiltered_candidates + filtered_candidates, gt_answer=gt_answer)
        else:
            # vote: [1, 2, 3] (pruning)
            return self._find_winner_filtered(prefiltered_candidates, filtered_candidates, gt_answer)


def main():
    parser = ArgumentParser()

    parser.add_argument("--generator", type=str, default="ours", choices=["ours", "fewshot_cot", "rap"])

    parser.add_argument("--note", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api", type=str, default="vllm")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--method", type=str, default='baseline', choices=['majvote', 'roundrobin', 'knockout', 'multi', 'baseline'])
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    parser.add_argument("--disable_save", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--threshold", type=float, default=0.999)

    # vLLM
    parser.add_argument("--max_num_seqs", type=int, default=256)

    # For multi-choice
    parser.add_argument("--multi_choice_prompt_type", type=str, default=None, choices=["fewshot", "instruct"])

    # For reasoning consistency
    parser.add_argument("--mask_left_boundary", type=float, default=0.2)
    parser.add_argument("--mask_right_boundary", type=float, default=0.5)
    parser.add_argument("--num_masked_solution_traces", type=int, default=4)
    parser.add_argument("--rc_mode", type=str, default='mid', choices=['loose', 'mid', 'strict', "maj"])
    parser.add_argument("--rc_temperature", type=float, default=0.8)
    parser.add_argument("--rc_n_completions", type=int, default=8)
    parser.add_argument("--save_inconsistent_candidates", action="store_true")
    parser.add_argument("--add_solution_trace", action="store_true")
    parser.add_argument("--add_terminal_trace", action="store_true")
    parser.add_argument("--special_parse_subq", action="store_true")
    parser.add_argument("--rc_criteria", type=str, default="freq")

    # For multi-discriminator
    parser.add_argument("--enable_multi_discriminator", action="store_true")
    parser.add_argument("--run_number", type=int, default=None)
    parser.add_argument("--partner_model_ckpt", type=str, default=None)

    # For caching
    parser.add_argument("--cache_root", type=str, default="cache")
    parser.add_argument("--cache_id", type=str, default="default")

    # For rollout
    parser.add_argument("--cutoff_rollout", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)

    args = parser.parse_args()

    if args.method == 'baseline':
        args.threshold = 0

    args.cache_dir = os.path.join(args.cache_root, args.cache_id)
    os.makedirs(args.cache_dir, exist_ok=True)
    args.cache_file = os.path.join(args.cache_dir, f"main_discriminator_cache.json")
    if args.run_number is not None:
        if args.run_number == 1:
            if os.path.exists(args.cache_file):
                print(f"Cache file already exists: {args.cache_file}. Exiting...")
                sys.exit(0)
            with open(os.path.join(args.cache_dir, "args.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
        elif args.run_number == 2:
            assert os.path.exists(args.cache_file), f"Cache file not found: {args.cache_file}"

    args.fewshot_config_path = os.path.join(args.prompt_dir, args.dataset_name, "fewshot_cot", "fewshot_cot_config.json")
    args.fewshot_prompt_path = os.path.join(args.prompt_dir, args.dataset_name, "fewshot_cot", "fewshot_cot_prompt.txt")

    fix_seeds(args.seed)
    print(args)

    answer_sheets_dir = os.path.join(args.root_dir, "answer_sheets")
    if args.resume:
        exp_id = args.resume
    else:
        exp_id = f"dis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}---{args.note}"

    discriminate_out_dir = os.path.join(args.root_dir, exp_id)
    if not args.disable_save:
        os.makedirs(discriminate_out_dir, exist_ok=True)
    args.discriminate_results_dir = os.path.join(discriminate_out_dir, "results")
    if not args.disable_save:
        os.makedirs(args.discriminate_results_dir, exist_ok=True)

    recording_file = os.path.join(discriminate_out_dir, "recording.json")

    recording = vars(args)

    evaluator = eval(f"{args.dataset_name}Evaluator()")
    if args.method in 'majvote':
        discriminator = MajorityVoteDiscriminator(args, evaluator)
    # elif args.method == 'roundrobin':
    #     discriminator = RoundRobinDiscriminator(args, evaluator)
    # elif args.method == 'knockout':
    #     discriminator = KnockoutDiscriminator(args, evaluator)
    # elif args.method == 'multi':
    #     assert args.run_number is not None
    #     assert args.partner_model_ckpt is not None
    #     discriminator = MultiDiscriminator(args, evaluator)
    else:
        discriminator = None

    #! ------ Select winner candidate for each example ------
    if args.generator == "ours":
        answer_sheet_json_files = [
            os.path.join(answer_sheets_dir, f) for f in os.listdir(answer_sheets_dir) if f.endswith("Answer.json")
        ]
        answer_sheet_json_files.sort()
        if args.start_idx > -1 and args.end_idx > -1:
            answer_sheet_json_files = answer_sheet_json_files[args.start_idx:args.end_idx]
    elif args.generator == "fewshot_cot":
        answer_sheet_json_files = [
            os.path.join(answer_sheets_dir, f) for f in os.listdir(answer_sheets_dir) if f.endswith(".json")
        ]
    elif args.generator == "rap":
        answer_sheets_dir = answer_sheets_dir.replace("answer_sheets", "")
        answer_sheet_json_files = [
            os.path.join(answer_sheets_dir, f) for f in os.listdir(answer_sheets_dir) if f.endswith(".json")
        ]

    num_correct, num_correct_majvote, num_correct_limit, num_tested = 0, 0, 0, 0
    terminal_trace_is_valid = True
    with tqdm(total=len(answer_sheet_json_files), disable=True) as pbar:
        total_num_candidates = 0
        for file_idx, answer_js_file in enumerate(answer_sheet_json_files):
            problem_id = int(answer_js_file.split("/")[-1].split(".")[0].replace(" - Answer", "").replace("Question ", ""))
            if args.resume and os.path.exists(os.path.join(args.discriminate_results_dir, f"problem-{problem_id}.json")):
                print(f"\n[Skip file {file_idx}; Total number of files: {len(answer_sheet_json_files)}]\n")
                with open(os.path.join(args.discriminate_results_dir, f"problem-{problem_id}.json"), "r") as f: temp_recording = json.load(f)
                correct = temp_recording["correct"]
                correct_majvote = temp_recording["correct_majvote"]
                correct_limit = temp_recording["correct_limit"]

                num_correct += int(correct)
                num_correct_majvote += int(correct_majvote)
                num_correct_limit += int(correct_limit)
                num_tested += 1

                info = f"Acc: {num_correct / num_tested:.4f}; Majority vote acc: {num_correct_majvote / num_tested:.4f}; Limit acc: {num_correct_limit / num_tested:.4f}"
                print(info)
                pbar.set_description(info, refresh=True)
            else:
                print(f"\n[Processing file {file_idx}; Total number of files: {len(answer_sheet_json_files)}]\n")
                try:
                    answer_js = read_json(answer_js_file)
                except:
                    continue

                try:
                    problem = answer_js["problem"]
                    # assert problem_id == answer_js["id"]
                    gold_answer = answer_js["gold_answer"]
                except:
                    pass

                if args.generator == "ours":
                    if args.add_solution_trace:
                        solutions_js = read_json(answer_js_file.replace("Answer", "Final Solutions"))
                        if args.cutoff_rollout > -1:
                            solutions_js = [s for s in solutions_js if s["rollout_id"] <= args.cutoff_rollout]
                    if args.add_terminal_trace and terminal_trace_is_valid:
                        try:
                            rollout_js = read_json(answer_js_file.replace("Answer", "Rollout Solutions"))
                        except:
                            terminal_trace_is_valid = False
                            print(f"Terminal trace not found for file {answer_js_file}. Skipping...")

                        if args.cutoff_rollout > -1:
                            rollout_js = [r for r in rollout_js if r["rollout_id"] <= args.cutoff_rollout]

                    # ------ Collect all_candidates, answer2candidates answer2confidence ------
                    all_candidates = []
                    solution_trace_dic = {}
                    if args.add_solution_trace:
                        for id, s in enumerate(solutions_js):
                            trace = s["trace"] if "trace" in s else s
                            solution_trace, final_step, _, reward = concat_solution_trace(trace)
                            if solution_trace in solution_trace_dic:
                                solution_trace_dic[solution_trace]["freq"] = solution_trace_dic[solution_trace]["freq"] + 1
                                solution_trace_dic[solution_trace]["reward"] = solution_trace_dic[solution_trace]["reward"] + reward
                                if len(solution_trace_dic[solution_trace]["final_step"]) < len(final_step):
                                    solution_trace_dic[solution_trace]["final_step"] = final_step
                            else:
                                solution_trace_dic[solution_trace] = {"freq": 1, "reward": reward, "final_step": final_step}
                            if args.special_parse_subq and len(trace) > 1 and len(trace[str(len(trace)-1)]["tot_step"]) == 0:
                                subq_suba_dic, _, _ = concat_subq_suba_trace(trace)
                                solution_trace_dic[solution_trace]["subq_suba_dic"] = subq_suba_dic
                    if args.add_terminal_trace and terminal_trace_is_valid:
                        for id, s in enumerate(rollout_js):
                            trace = s["trace"] if "trace" in s else s
                            solution_trace, final_step, _, reward = concat_solution_trace(trace)
                            if solution_trace in solution_trace_dic:
                                solution_trace_dic[solution_trace]["freq"] = solution_trace_dic[solution_trace]["freq"] + 1
                                solution_trace_dic[solution_trace]["reward"] = solution_trace_dic[solution_trace]["reward"] + reward
                                if len(solution_trace_dic[solution_trace]["final_step"]) < len(final_step):
                                    solution_trace_dic[solution_trace]["final_step"] = final_step
                            else:
                                solution_trace_dic[solution_trace] = {"freq": 1, "reward": reward, "final_step": final_step}
                            if args.special_parse_subq and len(trace) > 1 and len(trace[str(len(trace)-1)]["tot_step"]) == 0:
                                subq_suba_dic, _, _ = concat_subq_suba_trace(trace)
                                solution_trace_dic[solution_trace]["subq_suba_dic"] = subq_suba_dic
                    for solution_trace in solution_trace_dic.keys():
                        final_step = solution_trace_dic[solution_trace]["final_step"]
                        trace_freq = solution_trace_dic[solution_trace]["freq"]
                        trace_reward = solution_trace_dic[solution_trace]["reward"]
                        if args.special_parse_subq and "subq_suba_dic" in solution_trace_dic[solution_trace]:
                            masked_solution_trace_list = mask_subq_suba_trace(
                                solution_trace_dic[solution_trace]["subq_suba_dic"], num_return=args.num_masked_solution_traces, evaluator=evaluator
                            )
                            final_answer = evaluator.extract_answer_from_model_completion(final_step)
                            candidate = Candidate(solution_trace, deepcopy(masked_solution_trace_list), final_step, final_answer, id, trace_freq, trace_reward, c_type="subq")
                            all_candidates.append(candidate)
                        else:
                            masked_solution_trace_list = mask_solution_trace(
                                solution_trace, num_return=args.num_masked_solution_traces,
                                left_boundary=args.mask_left_boundary, right_boundary=args.mask_right_boundary
                            )
                            final_answer = evaluator.extract_answer_from_model_completion(final_step)
                            candidate = Candidate(solution_trace, deepcopy(masked_solution_trace_list), final_step, final_answer, id, trace_freq, trace_reward)
                            all_candidates.append(candidate)
                elif args.generator == "fewshot_cot":
                    vote_pool_dir = os.path.join(args.root_dir, "vote_pool")
                    file_id = answer_js_file.split("/")[-1].split(".")[0]
                    vote_pool_file = os.path.join(vote_pool_dir, f"{file_id}.json")
                    vote_pool = read_json(vote_pool_file)[:128]
                    all_candidates = []
                    for v_id, v in enumerate(vote_pool):
                        solution_trace = v
                        masked_solution_trace_list = mask_solution_trace(
                            solution_trace, num_return=args.num_masked_solution_traces,
                            left_boundary=args.mask_left_boundary, right_boundary=args.mask_right_boundary
                        )
                        final_step = v
                        final_answer = evaluator.extract_answer_from_model_completion(v)
                        candidate = Candidate(solution_trace, deepcopy(masked_solution_trace_list), final_step, final_answer, v_id)
                        all_candidates.append(candidate)
                elif args.generator == "rap":
                    file_id = answer_js_file.split("/")[-1].split(".")[0]
                    problem = answer_js[0]["question"]
                    gold_answer = answer_js[0]["answer"]
                    all_candidates = []
                    for v_id, v in enumerate(answer_js[:32]):
                        solution_trace, final_step = concat_rap_solution_trace(v["traj"])
                        masked_solution_trace_list = mask_solution_trace(
                            solution_trace, num_return=args.num_masked_solution_traces,
                            left_boundary=args.mask_left_boundary, right_boundary=args.mask_right_boundary
                        )
                        final_answer = evaluator.extract_answer_from_model_completion(final_step)
                        candidate = Candidate(solution_trace, deepcopy(masked_solution_trace_list), final_step, final_answer, v_id)
                        all_candidates.append(candidate)

                answer2candidates, answer2confidence, _ = group_candidates_by_answer(all_candidates, evaluator, args.rc_criteria)
                most_confident_answer = max(answer2candidates.keys(), key=lambda x: answer2confidence[x])
                highest_confidence = answer2confidence[most_confident_answer]
                assert highest_confidence > 0
                # -------------------------------------------------------------------------

                # candidates = [cands[0] for _, cands in answer2candidates.items()]   #! representative
                candidates = all_candidates  #! exhaustive
                total_num_candidates += len(candidates)

                if args.method == 'multi' and args.run_number == 1:
                    # ------ Filter answers by main discriminator ------
                    if not any(evaluator.check_answers_equiv(ans, gold_answer) for ans in answer2candidates.keys()):
                        print("Well, no correct answer in candidates. Skipping...")
                    else:
                        if highest_confidence > args.threshold:
                            print("You are very confident. Skipping...")
                        else:
                            discriminator.cache_main_discriminator_filter_results(candidates, problem, gold_answer, file_idx)

                    pbar.set_description(
                        f"File {file_idx} processed by main discriminator",
                        refresh=True,
                    )
                else:
                    # ------ Get winner answer ------
                    if not any(evaluator.check_answers_equiv(ans, gold_answer) for ans in answer2candidates.keys()):
                        # In this case, we know that there is no correct answer in the candidates
                        print("Well, no correct answer in candidates. Skipping...")
                        winner_answer = ""
                    else:
                        if highest_confidence > args.threshold:
                            print("You are very confident. Skipping...")
                            winner_answer = most_confident_answer
                        else:
                            winner_candidate = discriminator.select(problem, candidates, gt_answer=gold_answer, aux={"file_idx": file_idx, "problem_id": problem_id})
                            if winner_candidate is not None:
                                winner_answer = winner_candidate.final_answer
                            else:
                                winner_answer = most_confident_answer
                    # -------------------------------
                    correct = evaluator.check_answers_equiv(winner_answer, gold_answer)
                    correct_majvote = evaluator.check_answers_equiv(most_confident_answer, gold_answer)
                    correct_limit = 1 if any(evaluator.check_answers_equiv(ans, gold_answer) for ans in answer2candidates.keys()) else 0
                    print(f"==> Correct: {correct}")
                    try:
                        with open(os.path.join(args.discriminate_results_dir, f"problem-{problem_id}.json"), "r") as f: temp_recording = json.load(f)
                    except:
                        temp_recording = {}
                    temp_recording.update({
                        "correct": correct,
                        "correct_majvote": correct_majvote,
                        "correct_limit": correct_limit,
                    })
                    with open(os.path.join(args.discriminate_results_dir, f"problem-{problem_id}.json"), "w") as f: json.dump(temp_recording, f, indent=4)
                    num_correct += int(correct)
                    num_correct_majvote += int(correct_majvote)
                    num_correct_limit += int(correct_limit)
                    num_tested += 1

                    info = f"Acc: {num_correct / num_tested:.4f}; Majority vote acc: {num_correct_majvote / num_tested:.4f}; Limit acc: {num_correct_limit / num_tested:.4f}"
                    print(info)
                    pbar.set_description(info, refresh=True)

            pbar.update(1)
    #! --------------------------------------------------------

    if args.method == 'multi' and args.run_number == 1:
        discriminator.save_main_discriminator_filter_results()
    else:
        print(
            f"Accuracy: {num_correct / num_tested:.4f}; Majority vote accuracy: {num_correct_majvote / num_tested:.4f}; Limit accuracy: {num_correct_limit / num_tested:.4f}"
        )

        recording.update({
            "num_correct": num_correct,
            "num_correct_majvote": num_correct_majvote,
            "num_correct_limit": num_correct_limit,
            "num_tested": num_tested,
            "accuracy": num_correct / num_tested,
            "majority_vote_accuracy": num_correct_majvote / num_tested,
            "limit_accuracy": num_correct_limit / num_tested,
            "avg_num_candidates": total_num_candidates / num_tested,
        })

        print(f"Recording: \n{recording}")

        if not args.disable_save:
            with open(recording_file, "w") as f:
                json.dump(recording, f, indent=4)


if __name__ == "__main__":
    main()