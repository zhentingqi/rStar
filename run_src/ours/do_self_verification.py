import os
import json
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from argparse import ArgumentParser

class IOSystem:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.tokenizer, self.model = self.load_vLLM_model(model_path)
        self.answer_marker = "answer is"
        self.temperature = 0.8
        self.top_p = 0.95
        self.top_k = 40
        self.max_tokens = 256
        self.stop_tokens = ['\n']

    def load_vLLM_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(model=model_path, seed=1, trust_remote_code=True)
        return tokenizer, llm

    def generate_responses(self, model_input: List[str], num_return: int) -> List[List[str]]:
        vllm_response = generate_with_vLLM_model(
            self.model,
            input=model_input,
            temperature=self.temperature, 
            top_p=self.top_p,
            top_k=self.top_k,
            n=num_return,
            max_tokens=self.max_tokens,
            stop=self.stop_tokens
        )
        io_output_list = [[o.text for o in resp_to_single_input.outputs] for resp_to_single_input in vllm_response]
        return io_output_list

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except:
            pass
        try:
            import unicodedata
            res = unicodedata.numeric(s)
            return True, str(res)
        except:
            pass
        return False, None

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        is_number_a, number_a = self._is_number(answer_a)
        is_number_b, number_b = self._is_number(answer_b)
        if is_number_a and is_number_b:
            correct = number_a == number_b
        else:
            correct = False
        return correct

    def validate_completion(self, completion: str) -> bool:
        if self.answer_marker.lower() in completion.lower():
            return True
        return False


def generate_with_vLLM_model(
        model,
        input,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        n=1,
        max_tokens=256,
        logprobs=1,
        stop=['\n'],
        seed=1
):
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, top_k=top_k,
        n=n,
        logprobs=logprobs, max_tokens=max_tokens, stop=stop, seed=seed)
    output = model.generate(input, sampling_params, use_tqdm=False)
    return output


def qa_to_declarative(question: str, answer: str) -> str:
    system_prompt = "Please combine the Question and Answer into a complete declarative sentence.\n\n"
    examples = [
        "Question: How many books will they read in 5 weeks?\nAnswer: 15\nResponse: They will read 15 books in 5 weeks.",
        "Question: How many cupcakes do they sell in a week?\nAnswer: 840\nResponse: They sell 840 cupcakes in a week.",
        "Question: What is the total cost for 3.5 kilograms of apples if each kilogram costs $2.50?\nAnswer: 8.75\nResponse: The total cost for 3.5 kilograms of apples at $2.50 per kilogram is $8.75."
    ]
    question_input = f"Question: {question}\nAnswer: {answer}\nResponse:"
    return system_prompt + '\n\n'.join(examples) + '\n\n' + question_input


def is_logically_correct(statements: List[str], io_system: IOSystem, num_verification: int) -> Tuple[List[int], List[List[str]]]:
    system_prompt = "Given a Statement related to high school mathematics, please analyze and determine if the Statement is True or False. End your response with 'The answer is <True | False>'.\n\n"
    examples = [
        "Statement: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. The grove workers planted 4 trees today. Does it True or False?\nResponse: If the Grove workers will plant 4 trees today and there will be 21 trees after they are done. 21 - 4 = 17, there are 17 trees in the grove, but actually there are 15 trees, 17 != 15, which is different from the theme. The answer is False",
        "Statement: If there are 3 cars in the parking lot and 2 more cars arrive, There are 5 cars in the parking lot. Does it True or False?\nResponse: If there will be 5 cars in the parking lot, subtract 2 cars that will arrive, 5 - 2 = 3, so there are 2 cars in the parking lot, which is consistent with the theme. The answer is True",
        "Statement: Leah had 32 chocolates and her sister had 42. If they ate 35, they have 39 pieces left in total. Does it True or False?\nResponse: If there are 39 pieces of chocolates and 35 pieces of chocolate are eaten, Leah and her sister have 39 + 35 = 74 in total. Her sister's had 42, so Leah had 74 - 42 = 32, which is consistent with the theme. The answer is True",
        "Statement: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. Jason gave Denny 6 lollipops. Does it True or False?\nResponse: If Jason gave Denny 6 lollipops, and Jason now has 12 lollipops, so Jason originally had 6+12=18 lollipops, 18 != 20, which is different from the theme. The answer is False",
        "Statement: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. He has 9 toys now. Does it True or False?\nResponse: If Shawn now has 9 toys and his parents gave him two each, then he originally had 9 - 2 - 2 = 5, which is consistent with the theme. The answer is True.",
        "Statement: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. There are 18 computers in the server room. Does it True or False?\nResponse: Now there are 18 computers in the server room. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. So there were 18 - 20= -2 in the server room originally, -2 != 9, which is different from the theme. The answer is False.",
        "Statement: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. He had 40 golf balls at the end of Wednesday. Does it True or False?\nResponse: If Michael had 40 golf balls on Wednesday, he had 40+2=42 on Tuesday because he lost 2 golf balls on Wednesday. Due to losing 23 balls on Tuesday, he should have had 42+23=65 on Monday, but in fact Michael had 58 golf balls originally, which is different from the theme. The answer is False.",
        "Statement: Olivia has $23. She bought five bagels for $3 each. She has 8 dollars left. Does it True or False?\nResponse: If Olivia had $8 left and she bought five bagels for $3 each, so costs 5 * 3 = 15, so there was 8 + 15 = 23, which is consistent with the theme. The answer is True."
    ]
    prompts = [system_prompt + '\n\n'.join(examples) + f"\n\nStatement: {statement} Does it True or False?\nResponse:" for statement in statements]
    
    num_batches = (len(prompts) + 127) // 128
    cnts = []
    all_responses = []
    
    for i in range(num_batches):
        batch_prompts = prompts[i*128:(i+1)*128]
        batch_responses = io_system.generate_responses(batch_prompts, num_verification)
        
        for responses in batch_responses:
            cnt = sum(1 for response in responses if "answer is True" in response or "answer is true" in response)
            cnts.append(cnt)
            all_responses.append(responses)
    
    return cnts, all_responses


def self_verification(base_path: str, output_path: str, io_system: IOSystem, method: str = "maj_verify", num_verification: int = 10) -> None:
    os.makedirs(output_path, exist_ok=True)
    base_path = os.path.join(base_path, "answer_sheets")
    
    question_ids = set()
    for file_name in os.listdir(base_path):
        if file_name.endswith("Answer.json") or file_name.endswith("Sequential List.json"):
            question_id = file_name.split()[1]
            question_ids.add(question_id)
    
    results = {}
    correct_answers = 0
    total_questions = 0

    with tqdm(total=len(question_ids), desc="Processing questions") as pbar:
        for question_id in question_ids:
            answer_file = f"Question {question_id} - Answer.json"
            sequential_file = f"Question {question_id} - Sequential List.json"

            if not (os.path.exists(os.path.join(base_path, answer_file)) and os.path.exists(os.path.join(base_path, sequential_file))):
                continue

            with open(os.path.join(base_path, answer_file), 'r') as f:
                question_data = json.load(f)
            problem = question_data["problem"]
            try:
                if '. ' in problem:
                    condition, question = problem.rsplit('. ', 1)
                else:
                    condition, question = '', problem
            except:
                print(question_id)
                continue

            with open(os.path.join(base_path, sequential_file), 'r') as f:
                sequential_data = json.load(f)
            unique_answers = list(set(sequential_data["answer_list"]))
            ground_truth = sequential_data["ground_truth"]
            answer_counts = {answer: sequential_data["answer_list"].count(answer) for answer in unique_answers}

            answers_results = {}
            best_answer = None
            highest_score = -1
            verification_logs = {}

            declarative_prompts = [qa_to_declarative(question, answer) for answer in unique_answers]
            declaratives = io_system.generate_responses(declarative_prompts, 1)
            final_qas = [f"{condition}. {declarative[0].strip()}" for declarative in declaratives]

            true_counts, all_responses = is_logically_correct(final_qas, io_system, num_verification)

            for idx, answer in enumerate(unique_answers):
                verification_score = true_counts[idx] / num_verification
                num_answer_appear = answer_counts[answer]
                score = verification_score * num_answer_appear if method == "maj_verify" else verification_score
                answers_results[answer] = score
                verification_logs[answer] = all_responses[idx]

                if score > highest_score:
                    highest_score = score
                    best_answer = answer

            final_result = {
                "unique_answers": answers_results,
                "ground_truth": ground_truth,
                "final_answer": best_answer,
                "correct": io_system.check_answers_equiv(ground_truth, best_answer),
                "verification_logs": verification_logs
            }
            results[question_id] = final_result

            total_questions += 1
            if final_result["correct"]:
                correct_answers += 1

            accuracy = correct_answers / total_questions
            pbar.set_postfix({"accuracy": f"{accuracy * 100:.2f}%"})
            pbar.update(1)

            # with open(os.path.join(output_path, f"{question_id}.json"), 'w') as f:
            #     json.dump(final_result, f, indent=2)

    with open(os.path.join(output_path, "final_accuracy.json"), 'w') as f:
        json.dump({"accuracy": accuracy}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--exp_dir_path", type=str, required=True)
    parser.add_argument("--discriminator_dir", type=str, default="../Phi-3-mini-4k-instruct_folio")
    parser.add_argument("--num_verification", type=int, default=10)
    args = parser.parse_args()
    
    method = "maj_verify"
    discriminator_dir = args.discriminator_dir
    num_verification = args.num_verification
    output_path = os.path.join(args.exp_dir_path, f"verification_results_{num_verification}")

    io_system = IOSystem(discriminator_dir)

    self_verification(args.exp_dir_path, output_path, io_system, method, num_verification)
