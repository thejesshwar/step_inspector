import json
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/assignment3_data.json"
HIDDEN_DIM = 896
NUM_LAYERS = 24

def parse_steps(candidate_text):
    """
    Verified on 650/650 examples.
    Extracts character start and end indices for each numbered step.
    """
    pattern = re.compile(r"(Step\s+\d+[:\.\-].*?)(?=(?:\n\s*Step\s+\d+[:\.\-]|$))", re.DOTALL)
    matches = list(pattern.finditer(candidate_text))
    return [(m.start(), m.end()) for m in matches]

def extract_split(split_data, model, tokenizer, device, max_steps, split_name="split"):
    step_embeddings_mean = []
    step_embeddings_last = []
    step_targets = []
    step_masks = []
    gold_first_errors = []
    step_token_lengths = []

    print(f"\nProcessing '{split_name}' split ({len(split_data)} examples, padded to max_steps={max_steps})")

    for i, ex in enumerate(tqdm(split_data)):
        question = ex["question"]
        candidate = ex["candidate"]
        verdict = ex.get("verdict", "CORRECT")
        first_error = ex.get("first_error", None)

        gold_first_error = 0 if (first_error is None or verdict == "CORRECT") else int(first_error)

        # Native Qwen Chat Template
        messages = [
            {"role": "system", "content": "You solve arithmetic word problems accurately."},
            {"role": "user", "content": f"Question:\n{question}\n\nCandidate solution:\n{candidate}"}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize
        enc = tokenizer(full_prompt, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = torch.tensor([enc["input_ids"]], device=device)
        offset_mapping = enc["offset_mapping"]

        # Robust substring location
        candidate_start = full_prompt.rfind(candidate)
        assert candidate_start != -1, f"Candidate string not found in prompt for {ex['source_id']}"
        step_spans = parse_steps(candidate)
        assert len(step_spans) == ex["num_steps"], f"Step count mismatch in {ex['source_id']}"

        # Map character spans to token indices
        step_token_spans = []
        for s_start, s_end in step_spans:
            abs_start = candidate_start + s_start
            abs_end = candidate_start + s_end

            tok_indices = [
                idx for idx, (ts, te) in enumerate(offset_mapping)
                if ts < abs_end and te > abs_start and (ts != te)  # exclude zero-width special tokens
            ]
            assert len(tok_indices) > 0, f"Step with 0 tokens detected in {ex['source_id']}"
            step_token_spans.append((tok_indices[0], tok_indices[-1] + 1))

        num_steps = len(step_token_spans)

        # Forward pass through frozen Qwen
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            # Stack all 24 transformer layers -> shape: (24, Seq_Len, 896)
            all_layers = torch.stack(outputs.hidden_states[1:], dim=0)[:, 0, :, :]

        step_means = []
        step_lasts = []
        step_lens = []

        for t_start, t_end in step_token_spans:
            step_slice = all_layers[:, t_start:t_end, :]       # (24, tokens, 896)
            step_means.append(step_slice.mean(dim=1))          # (24, 896) - Mean pooling
            step_lasts.append(all_layers[:, t_end - 1, :])      # (24, 896) - Last token boundary
            step_lens.append(t_end - t_start)

        step_means = torch.stack(step_means, dim=1)  # (24, num_steps, 896)
        step_lasts = torch.stack(step_lasts, dim=1)  # (24, num_steps, 896)

        # Pad to dynamic max_steps
        padded_mean = torch.zeros((NUM_LAYERS, max_steps, HIDDEN_DIM), dtype=torch.float16)
        padded_last = torch.zeros((NUM_LAYERS, max_steps, HIDDEN_DIM), dtype=torch.float16)

        padded_mean[:, :num_steps, :] = step_means.half().cpu()
        padded_last[:, :num_steps, :] = step_lasts.half().cpu()

        targets = torch.zeros(max_steps, dtype=torch.float32)
        masks = torch.zeros(max_steps, dtype=torch.float32)
        lens = torch.zeros(max_steps, dtype=torch.long)

        lens[:num_steps] = torch.tensor(step_lens)

        # Causal Loss Masking Rule
        if gold_first_error == 0:
            masks[:num_steps] = 1.0
        else:
            k = gold_first_error
            masks[:k] = 1.0
            targets[k - 1] = 1.0  # Exactly the first error step gets 1.0

        step_embeddings_mean.append(padded_mean)
        step_embeddings_last.append(padded_last)
        step_targets.append(targets)
        step_masks.append(masks)
        gold_first_errors.append(gold_first_error)
        step_token_lengths.append(lens)

        # Periodic VRAM garbage collection to prevent Windows CUDA fragmentation
        if (i + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    stacked_mean = torch.stack(step_embeddings_mean) # (N, 24, max_steps, 896)
    stacked_last = torch.stack(step_embeddings_last) # (N, 24, max_steps, 896)

    return {
        "step_embeddings_mean": stacked_mean,
        "step_embeddings_last": stacked_last,
        "step_embeddings": stacked_mean,                 # Memory alias
        "step_targets": torch.stack(step_targets),        # (N, max_steps)
        "step_masks": torch.stack(step_masks),            # (N, max_steps)
        "gold_first_errors": torch.tensor(gold_first_errors, dtype=torch.long), # (N,)
        "step_token_lengths": torch.stack(step_token_lengths) # (N, max_steps)
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    # Dynamically find the absolute maximum steps in the dataset
    max_steps = max(ex["num_steps"] for split in splits.values() for ex in split)
    print(f"Maximum reasoning steps found in dataset: {max_steps}")

    os.makedirs("data", exist_ok=True)

    # 1. Process Train (500)
    train_cache = extract_split(splits["train"], model, tokenizer, device, max_steps, "train")
    torch.save(train_cache, "data/train_cached.pt")
    print("Saved data/train_cached.pt")

    # 2. Process Validation (50)
    val_cache = extract_split(splits["validation"], model, tokenizer, device, max_steps, "validation")
    torch.save(val_cache, "data/val_cached.pt")
    print("Saved data/val_cached.pt")

    # 3. Process Test (100)
    test_cache = extract_split(splits["test"], model, tokenizer, device, max_steps, "test")
    torch.save(test_cache, "data/test_cached.pt")
    print("Saved data/test_cached.pt")

    print("\nExtraction complete. All tensors are cached and follow the Tensor Contract.")

if __name__ == "__main__":
    main()