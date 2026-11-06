import torch
from pathlib import Path
import os
import re
import pandas as pd
from transformers import pipeline

torch.manual_seed(43)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(43)

MODEL_ID = "google/medgemma-4b-it"
MODEL_ID = os.getenv("MODEL_ID", MODEL_ID)

INPUT_PATH = "data/tr_medqa_mcq_with_disease_mentions.json"
INPUT_PATH = os.getenv("INPUT_PATH", INPUT_PATH)

pipe = pipeline(
    "image-text-to-text",
    model=MODEL_ID,
    torch_dtype=torch.bfloat16,
    device="cuda" if torch.cuda.is_available() else "cpu",
    token=os.getenv("HF_TOKEN"),
)

input_data = pd.read_json(INPUT_PATH, lines=True)

output_path = f"results/{MODEL_ID.split('/')[-1]}/{Path(INPUT_PATH).stem}_and_answers.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

answer_re = re.compile(r"Final Answer:\s*\(?([A-D])\)?", re.IGNORECASE)
def extract_answer(text):
    match = answer_re.search(text)
    return match.group(1).upper() if match else None

if os.path.exists(output_path):
    done_so_far = pd.read_json(output_path, lines=True)
    n_done = len(done_so_far)
    raw_predictions = done_so_far["raw_prediction"].tolist()
    predicted_answers = done_so_far["predicted_answer"].tolist()
    print(f"resuming from {output_path}, {n_done} samples already done")
else:
    n_done = 0
    raw_predictions = []
    predicted_answers = []

for n, prompt in enumerate(input_data["prompt"], start=1):
    if n <= n_done:
        continue
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    output = pipe(text=messages, max_new_tokens=2000, do_sample=False)
    raw_predictions.append(output[0]["generated_text"][-1]["content"])
    predicted_answers.append(extract_answer(raw_predictions[-1]))

    if n % 10 == 0 or n == len(input_data):
        done_so_far = input_data.iloc[:len(raw_predictions)].copy()
        done_so_far["raw_prediction"] = raw_predictions
        done_so_far["predicted_answer"] = predicted_answers
        done_so_far.to_json(output_path, orient="records", lines=True, force_ascii=False)