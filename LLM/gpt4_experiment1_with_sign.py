import openai
import os
import json
import argparse
import base64
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import re

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=50))
def get_gpt_response(messages, temperature=0.5):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_image_message(image_path, text):
    base64_image = encode_image(image_path)
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]

def run_instructions(instructions):
    results = []
    for instruction in instructions:
        messages = [
            {"role": "system", "content": "Please answer the following question based on the instructions provided."},
            {"role": "user", "content": instruction['preamble']},
            {"role": "user", "content": instruction['question']}
        ]
        response = get_gpt_response(messages)
        results.append({
            "instruction": instruction['preamble'],
            "question": instruction['question'],
            "response": response
        })
    return results

def parse_moral_judgment(response):
    match = re.search(r'\[value\]\s*(-?\d+)\s*\[/value\]', response)
    if match:
        value = int(match.group(1))
        return value if 0 <= value <= 100 else None
    return None

def get_valid_response(messages, temperature=0.5):
    while True:
        response = get_gpt_response(messages, temperature)
        moral_judgment = parse_moral_judgment(response)
        if moral_judgment is not None:
            return response, moral_judgment

def run_trial(trial, context):
    results = []
    # system_prompt = "Answer the following questions based on the information provided. Give your best judgment even if the information is incomplete. Verbalize your step by step reasoning and thinking process. Then, provide the number in this format: [value] number [/value]. For example: [value] 42 [/value]."
    system_prompt = "Answer the following questions based on the information provided. Do not make any excuses about not having enough information or some other reason for not being able to answer. Based on your best judgment, provide only provide the number as your answer in this format: [value] number [/value]. For example: [value] 42 [/value]. Do not include any other text or explanation."


    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ]
    
    if 'image' in trial:
        image_path = os.path.join(SCRIPT_DIR, 'experiment_stimuli', 'exp1_stimuli_with_sign', trial['image'])
        base_messages.append({"role": "user", "content": create_image_message(image_path, trial['description'])})
    
    for question in tqdm(trial['questions'], desc="Processing questions"):
        messages = base_messages.copy()  # Reset messages for each question
        messages.append({"role": "user", "content": question})
        responses = []
        moral_judgments = []
        
        for _ in tqdm(range(50), desc="Sampling responses"):
            response, moral_judgment = get_valid_response(messages, temperature=1.0)
            responses.append(response)
            moral_judgments.append(moral_judgment)
        
        results.append({
            "question": question,
            "responses": responses,
            "moral_judgments": moral_judgments
        })
    
    return results, system_prompt

def process_experiment(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        experiment_data = json.load(f)
    
    results = {
        "instructions": run_instructions(experiment_data['instructions']),
        "trials": [],
        "system_prompt": ""
    }
    
    context = "\n".join([instr['preamble'] for instr in experiment_data['instructions']])
    
    for trial in tqdm(experiment_data['trials'], desc="Processing trials"):
        trial_results, system_prompt = run_trial(trial, context)
        results['trials'].append({
            "trial_info": trial,
            "results": trial_results
        })
        results['system_prompt'] = system_prompt

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Run GPT-4 experiment.")
    parser.add_argument(
        "input_file", 
        type=str, 
        nargs='?', 
        default=os.path.join(SCRIPT_DIR, "experiment_stimuli", "exp1_stimuli_with_sign", "experiment_config.json"), 
        help="Path to the input JSON file containing experiment configuration."
    )
    parser.add_argument(
        "output_file", 
        type=str, 
        nargs='?', 
        default=os.path.join(SCRIPT_DIR, "experiment_results", "gpt4_experiment1_with_sign", "experiment_results.json"), 
        help="Path to the output JSON file for experiment results."
    )
    args = parser.parse_args()

    process_experiment(args.input_file, args.output_file)

if __name__ == "__main__":
    main()