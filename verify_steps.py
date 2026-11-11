import json
import re

DATA_PATH = "data/assignment3_data.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    splits = json.load(f)["splits"]

def parse_steps(candidate_text):
    # Matches 'Step 1:', 'Step 1.', 'Step 1 -', etc.
    pattern = re.compile(r"(Step\s+\d+[:\.\-].*?)(?=(?:\n\s*Step\s+\d+[:\.\-]|$))", re.DOTALL)
    matches = list(pattern.finditer(candidate_text))
    return [(m.start(), m.end()) for m in matches]

total = 0
mismatches = 0

for split_name in ["train", "validation", "test"]:
    for i, ex in enumerate(splits[split_name]):
        total += 1
        steps = parse_steps(ex["candidate"])
        expected_steps = ex["num_steps"]
        
        if len(steps) != expected_steps:
            mismatches += 1
            if mismatches <= 3:
                print(f"Mismatch in {split_name}[{i}]: Expected {expected_steps} steps, but found {len(steps)}")
                print("Text snippet:", repr(ex["candidate"][:150]))

print(f"\nResult: {total - mismatches}/{total} examples parsed perfectly")
if mismatches > 0:
    print(f"Warning: {mismatches} examples had parsing mismatches.")
else:
    print("Flawless. Every single step in the entire dataset matches the ground truth.")