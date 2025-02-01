import numpy as np
from openai import OpenAI
import json
from .test_suites import instance_loc_table, test_code as common_test_code, test_code_for_behind, test_code_for_between, test_code_for_front, test_code_for_unary
from pathlib import Path
from copy import deepcopy
import random
import os
import traceback

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
sample_num = 20

running_list = [
    ("left", common_test_code, "on the left"),
    ("right", common_test_code, "on the right"),
    ("between", test_code_for_between, "between"),
    ("corner", test_code_for_unary, "at the corner"),
    ("above", common_test_code, "on top of"),
    ("below", common_test_code, "below"),
    ("behind", common_test_code, "behind"),
]

def complete(messages):
    body = {
        "messages": messages,
        "model": "gpt-4o-2024-08-06",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,
    }
    response = client.chat.completions.create(
        **body,
        seed=42,
        n=sample_num
    )
    
    return response

def vectorize_code(code):
    
    system_message = """You are an programming expert. Your task is to vectorize the functions in a class, while keep all variables and functions in the class.
Wrap your code with ```python and ```."""
    user_message = f"""The class is defined as follows:\n{code}"""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    body = {
        "messages": messages,
        "model": "gpt-4o-2024-08-06",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,
    }
    response = client.chat.completions.create(
        **body,
        seed=42,
        n=1
    )
    return response.choices[0].message.content.split("```python")[1].split("```")[0]

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()


def main(relation_name, function_, prompt):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    base = Path(f"data/test_data/{relation_name}")
    traj_base = base / "trajs"
    if not os.path.exists(traj_base):
        os.makedirs(traj_base, exist_ok=True)
    initial_system = file_to_string(base / "prompt.txt")
    feedback = file_to_string("data/test_data/feedback.txt")
    test_case = file_to_string(base / "feedback_template.txt")

    init_messages = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": f'Your task is write a class about relation "{prompt}".'}
    ]
    
    max_iteration = 5
    # first iteration
    messages = deepcopy(init_messages)
    responses = complete(messages)
    total_prompt_tokens += responses.usage.prompt_tokens
    total_completion_tokens += responses.usage.completion_tokens
    print("responses done.")
    idx_within_iteration = 0
    for choice in responses.choices:
        content = choice.message.content
        try:
            dir = traj_base / f"iter_0" / \
                f"choice_{idx_within_iteration}"
            os.makedirs(dir, exist_ok=True)
            with open(dir / "raw_response.txt", "w") as f:
                f.write(content)
            try:
                code = content.split("```python")[1].split("```")[0]
            except IndexError:
                print(content)
            code = vectorize_code(code)
            
            with open(dir / "concept.py", "w") as f:
                f.write(code)
            idx_within_iteration += 1
            
            print("Running test for iter 0, choice", idx_within_iteration)
            try:
                correct, total, failure_cases = function_(relation_name, dir / "concept.py")
            except Exception as e:
                print(traceback.format_exc())
                correct, total, failure_cases = 0, 1, []
            
            json.dump(failure_cases, open(
                dir / "failure_cases.json", "w"), indent=4)
            json.dump(
                {
                    "correct": correct,
                    "total": total,
                }, open(dir / "result.json", "w"), indent=4)
            print(
                f"Iteration 0, Choice {idx_within_iteration}, Correct: {correct}, Total: {total}, Success Rate: {correct / total}")
        
        except Exception as e:
            print(traceback.format_exc())
            continue
        
        if correct == total:
            print("price is $", total_prompt_tokens / 1e6 * 2.5 + total_completion_tokens / 1e6 * 10)
            print("done.")
            exit()
        
        print("price is $", total_prompt_tokens / 1e6 * 2.5 + total_completion_tokens / 1e6 * 10)
    
    for i in range(1, max_iteration):
        print(f"Starting iteration {i}")
        last_iteration_dir = traj_base / f"iter_{i-1}"
        idx_within_iteration = 0
        messages_acc = []
        max_acc = 0
        for choice in os.listdir(last_iteration_dir):
            choice_dir = last_iteration_dir / choice
            result = json.load(open(choice_dir / "result.json"))
            acc = result["correct"] / result["total"]
            max_acc = max(max_acc, acc)
        for choice in os.listdir(last_iteration_dir):
            messages = deepcopy(init_messages)
            choice_dir = last_iteration_dir / choice
            result = json.load(open(choice_dir / "result.json"))
            acc = result["correct"] / result["total"]
            if acc == 0 and max_acc > 0:
                continue
            force_rewrite = ""
            if acc < 0.6: 
                force_rewrite = '''After test, the pass rate of your code is too low. So you MUST check carefully where the problem is. If you can't find the problem, you should
come up with a new algorithm and re-write your code. 
Don't forget the following tips:
(1) You should imagine that you are at position (0, 0, 0) to determine the relative positions.
(2) Remember you are **in** the scene and look around, not look from the top. So never use the same way as 2D environment.
(3) Don't use any of x-axis or y-axis as your perspective, Your method should work for every perspective.
(4) The horizontal plane is x-y plane.'''
            content = file_to_string(choice_dir / "raw_response.txt")
            try:
                code = content.split("```python")[1].split("```")[0]
            except IndexError:
                continue
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            failure_cases = json.load(open(choice_dir / "failure_cases.json"))
            failure_cases = random.sample(failure_cases, min(3, len(failure_cases)))
            cases = ""
            for idx, case in enumerate(failure_cases):
                scan_id = case["scan_id"]
                instance_locs = instance_loc_table[scan_id]
                cases += test_case.format(
                    idx=idx + 1,
                    relation=relation_name,
                    target_loc = instance_locs[case["target"]],
                    anchor_loc = instance_locs[case["target_anchor"]],
                    pred_loc = instance_locs[case["pred"]],
                    pred_anchor_loc = instance_locs[case["pred_anchor"]]
                )
            user_message = {"role": 'user', "content": feedback.format(test_cases=cases, force_rewrite=force_rewrite,)}
            messages.append(user_message)
            assert len(messages) == 4
            messages_acc.append((messages, acc))
        messages_acc = sorted(messages_acc, key=lambda x: x[1], reverse=True)
        messages_acc = messages_acc[:3]
        print("getting response for iter", i)
        for messages, _ in messages_acc:
            responses = complete(messages)
            total_prompt_tokens += responses.usage.prompt_tokens
            total_completion_tokens += responses.usage.completion_tokens
            for choice in responses.choices:
                content = choice.message.content
                try:
                    code = content.split("```python")[1].split("```")[0]
                    code = vectorize_code(code)
                    dir = traj_base / f"iter_{i}" / \
                        f"choice_{idx_within_iteration}"
                    os.makedirs(dir, exist_ok=True)
                    with open(dir / "raw_response.txt", "w") as f:
                        f.write(content)
                    with open(dir / "concept.py", "w") as f:
                        f.write(code)
                    print(f"Running test for iter {i}, choice", idx_within_iteration)
                    try:
                        correct, total, failure_cases = function_(relation_name, dir / "concept.py")
                    except Exception:
                        print(traceback.format_exc())
                        correct, total, failure_cases = 0, 1, []
                    json.dump(failure_cases, open(
                        dir / "failure_cases.json", "w"), indent=4)
                    json.dump(
                        {
                            "correct": correct,
                            "total": total,
                        }, open(dir / "result.json", "w"), indent=4)
                    print(
                        f"Iteration {i}, Choice {idx_within_iteration}, Correct: {correct}, Total: {total}, Success Rate: {correct / total}")
                    
                except Exception as e:
                    print(traceback.format_exc())
                    continue
                idx_within_iteration += 1
                if correct == total:
                    print("done.")
                    exit()
        print("price is $", total_prompt_tokens / 1e6 * 2.5 + total_completion_tokens / 1e6 * 10)
    return total_prompt_tokens, total_completion_tokens

if __name__ == "__main__":
    for relation_name, function_, prompt in running_list:
        main(relation_name, function_, prompt)
