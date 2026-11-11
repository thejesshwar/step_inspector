import json

DATA_PATH = "data/assignment3_data.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

splits = raw_data["splits"]
print("Splits found:", list(splits.keys()))

for split_name, split_list in splits.items():
    print(f"  - Split '{split_name}': {len(split_list)} items")

# Peek at the first example in 'train'
first_split = list(splits.keys())[0]
sample = splits[first_split][0]

print(f"\nSample item from '{first_split}'")
for k, v in sample.items():
    if isinstance(v, str) and len(v) > 150:
        print(f"  {k}: {repr(v[:150])} [truncated]")
    else:
        print(f"  {k}: {v}")