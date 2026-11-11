"""
extract_and_cache.py
--------------------
Course: Foundation of Large Language Models (AI60213)
Project: Step-INSPECTOR
Track: Member 3 (Causal Masking & Loss Dynamics) & Member 1

Runs the frozen base model (Qwen2.5-0.5B-Instruct), extracts step representations
across all 24 layers using tokenizer offset mappings, and applies the causal loss
supervision mask to prevent cascading error noise.
"""

import json
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from spatial_pooling import StepSpanExtractor, MeanStepPooling, LastTokenStepPooling

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/assignment3_data.json"
HIDDEN_DIM = 896
NUM_LAYERS = 24

def extract_split(split_data: list, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                  device: str, max_steps: int, split_name: str = "split") -> dict:
    """
    Extracts multi layer hidden states and applies Causal Loss Masking.
    Caches tensors in float16 to keep disk footprint under 400 MB.
    """
    step_embeddings_mean = []
    step_embeddings_last = []
    step_targets = []
    step_masks = []
    gold_first_errors = []
    step_token_lengths = []

    mean_pooler = MeanStepPooling()
    last_pooler = LastTokenStepPooling()

    print(f"\nProcessing '{split_name}' split ({len(split_data)} examples, padded to max_steps={max_steps})")

    for i, ex in enumerate(tqdm(split_data)):
        verdict = ex.get("verdict", "CORRECT")
        first_error = ex.get("first_error", None)
        
        # 0 = CORRECT; 1..M = step index of earliest substantive error
        gold_first_error = 0 if (first_error is None or verdict == "CORRECT") else int(first_error)

        messages = [
            {"role": "system", "content": "You solve arithmetic word problems accurately."},
            {"role": "user", "content": f"Question:\n{ex['question']}\n\nCandidate solution:\n{ex['candidate']}"}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(full_prompt, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = torch.tensor([enc["input_ids"]], device=device)
        
        # Exact substring location with assertion
        try:
            candidate_start = full_prompt.index(ex['candidate'])
        except ValueError:
            candidate_start = full_prompt.rfind(ex['candidate'])
            if candidate_start == -1:
                raise ValueError(f"Candidate text not found in chat prompt for item {ex.get('source_id', i)}")
        
        extractor = StepSpanExtractor(
            candidate_text=ex['candidate'], 
            offset_mapping=enc["offset_mapping"], 
            candidate_start=candidate_start
        )
        num_steps = len(extractor)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            # outputs.hidden_states[0] is embedding; [1:] are Transformer layers 1 to 24
            all_layers = torch.stack(outputs.hidden_states[1:], dim=0)[:, 0, :, :] # (24, Seq_Len, 896)

        step_means = []
        step_lasts = []
        step_lens = []

        for span in extractor:
            step_slice = all_layers[:, span.tok_start:span.tok_end, :] # (24, Step_Tokens, 896)
            
            step_means.append(mean_pooler(step_slice)) 
            step_lasts.append(last_pooler(step_slice))
            step_lens.append(span.num_tokens)

        step_means = torch.stack(step_means, dim=1)  # (24, num_steps, 896)
        step_lasts = torch.stack(step_lasts, dim=1)  # (24, num_steps, 896)

        # Pre allocate padded tensors in float16 on CPU
        padded_mean = torch.zeros((NUM_LAYERS, max_steps, HIDDEN_DIM), dtype=torch.float16)
        padded_last = torch.zeros((NUM_LAYERS, max_steps, HIDDEN_DIM), dtype=torch.float16)

        padded_mean[:, :num_steps, :] = step_means.half().cpu()
        padded_last[:, :num_steps, :] = step_lasts.half().cpu()

        targets = torch.zeros(max_steps, dtype=torch.float32)
        masks = torch.zeros(max_steps, dtype=torch.float32)
        lens = torch.zeros(max_steps, dtype=torch.long)

        lens[:num_steps] = torch.tensor(step_lens, dtype=torch.long)
        # CAUSAL LOSS MASKING
        # If solution is CORRECT: Supervise all steps as 0.
        # If solution is INCORRECT at step k: Supervise steps 1..k.
        # Steps > k are masked out to avoid learning on corrupted premises.
        if gold_first_error == 0:
            masks[:num_steps] = 1.0
        else:
            k = gold_first_error
            masks[:k] = 1.0
            targets[k - 1] = 1.0  # Exactly step k receives label 1.0

        step_embeddings_mean.append(padded_mean)
        step_embeddings_last.append(padded_last)
        step_targets.append(targets)
        step_masks.append(masks)
        gold_first_errors.append(gold_first_error)
        step_token_lengths.append(lens)

        # Regular memory reclamation
        if (i + 1) % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    stacked_mean = torch.stack(step_embeddings_mean)
    
    return {
        "step_embeddings_mean": stacked_mean,
        "step_embeddings_last": torch.stack(step_embeddings_last),
        "step_embeddings": stacked_mean,
        "step_targets": torch.stack(step_targets),
        "step_masks": torch.stack(step_masks),
        "gold_first_errors": torch.tensor(gold_first_errors, dtype=torch.long),
        "step_token_lengths": torch.stack(step_token_lengths)
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {MODEL_ID} on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map=device
    ).eval()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing data file at '{DATA_PATH}'. Place assignment3_data.json there.")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    # Calculate global max_steps across all 3 splits for consistent padding
    max_steps = max(ex["num_steps"] for split in splits.values() for ex in split)
    print(f"Maximum reasoning steps across dataset: {max_steps}")

    os.makedirs("data", exist_ok=True)

    # 1. Train Split (500)
    train_cache = extract_split(splits["train"], model, tokenizer, device, max_steps, "train")
    torch.save(train_cache, "data/train_cached.pt")
    print("Saved -> data/train_cached.pt")

    # 2. Validation Split (50)
    val_cache = extract_split(splits["validation"], model, tokenizer, device, max_steps, "validation")
    torch.save(val_cache, "data/val_cached.pt")
    print("Saved -> data/val_cached.pt")

    # 3. Test Split (100)
    test_cache = extract_split(splits["test"], model, tokenizer, device, max_steps, "test")
    torch.save(test_cache, "data/test_cached.pt")
    print("Saved -> data/test_cached.pt")

    print("\nSUCCESS All multi layer tensor representations extracted and cached.")

if __name__ == "__main__":
    main()