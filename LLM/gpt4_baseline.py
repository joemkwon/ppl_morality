"""Exploratory GPT-4o baseline for the rule-interpretation task.

NOT PART OF THE ACCEPTED CogSci 2025 PAPER. This is kept for transparency and
future work only -- no LLM result appears in the paper. See ``LLM/README.md``.

Consolidates the four original near-identical scripts
(``gpt4_experiment{1,2}_{with,without}_sign.py``) into one parametrized entry
point. Behaviour is unchanged: GPT-4o is shown the map stimulus, asked each
question 50x at temperature 1.0, and the ``[value]N[/value]`` answer is parsed.

Usage:
    export OPENAI_API_KEY=...
    python gpt4_baseline.py --experiment 1 --sign        # posted-sign condition
    python gpt4_baseline.py --experiment 2 --no-sign      # no-sign condition
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path

import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "Answer the following questions based on the information provided. Do not "
    "make any excuses about not having enough information or some other reason "
    "for not being able to answer. Based on your best judgment, provide only "
    "the number as your answer in this format: [value] number [/value]. For "
    "example: [value] 42 [/value]. Do not include any other text or explanation."
)
NUM_SAMPLES = 50

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=50))
def get_gpt_response(messages, temperature: float = 0.5) -> str:
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=temperature
    )
    return response.choices[0].message.content.strip()


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def parse_moral_judgment(response: str) -> int | None:
    match = re.search(r"\[value\]\s*(-?\d+)\s*\[/value\]", response)
    if match:
        value = int(match.group(1))
        return value if 0 <= value <= 100 else None
    return None


def get_valid_response(messages, temperature: float = 1.0) -> tuple[str, int]:
    while True:
        response = get_gpt_response(messages, temperature)
        judgment = parse_moral_judgment(response)
        if judgment is not None:
            return response, judgment


def run_instructions(instructions: list[dict]) -> list[dict]:
    results = []
    for instruction in instructions:
        messages = [
            {
                "role": "system",
                "content": "Please answer the following question "
                "based on the instructions provided.",
            },
            {"role": "user", "content": instruction["preamble"]},
            {"role": "user", "content": instruction["question"]},
        ]
        results.append(
            {
                "instruction": instruction["preamble"],
                "question": instruction["question"],
                "response": get_gpt_response(messages),
            }
        )
    return results


def run_trial(trial: dict, context: str, stimuli_dir: Path) -> list[dict]:
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    if "image" in trial:
        image_b64 = encode_image(stimuli_dir / trial["image"])
        base_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": trial["description"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        )

    results = []
    for question in tqdm(trial["questions"], desc="questions"):
        messages = base_messages + [{"role": "user", "content": question}]
        responses, judgments = [], []
        for _ in tqdm(range(NUM_SAMPLES), desc="samples", leave=False):
            response, judgment = get_valid_response(messages, temperature=1.0)
            responses.append(response)
            judgments.append(judgment)
        results.append({"question": question, "responses": responses, "moral_judgments": judgments})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--experiment", type=int, choices=[1, 2], required=True)
    sign = ap.add_mutually_exclusive_group(required=True)
    sign.add_argument("--sign", action="store_true", help="posted-sign condition")
    sign.add_argument("--no-sign", action="store_true", help="no-sign condition")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    suffix = "with_sign" if args.sign else "no_sign"
    stimuli_dir = SCRIPT_DIR / "experiment_stimuli" / f"exp{args.experiment}_stimuli_{suffix}"
    out_path = (
        Path(args.out)
        if args.out
        else (SCRIPT_DIR / "experiment_results" / f"gpt4_experiment{args.experiment}_{suffix}.json")
    )

    config = json.loads((stimuli_dir / "experiment_config.json").read_text())
    context = "\n".join(i["preamble"] for i in config["instructions"])
    results = {
        "instructions": run_instructions(config["instructions"]),
        "system_prompt": SYSTEM_PROMPT,
        "trials": [
            {"trial_info": trial, "results": run_trial(trial, context, stimuli_dir)}
            for trial in tqdm(config["trials"], desc="trials")
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
