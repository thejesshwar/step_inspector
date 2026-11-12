"""
extract_and_cache.py
Course: Foundation of Large Language Models (AI60213)
Project: Step-INSPECTOR
Track: Member 3 (Causal Masking & Loss Dynamics) & Member 1

Runs the frozen base model (Qwen/Qwen3-0.6B), extracts step representations
across all 28 layers using tokenizer offset mappings, and applies the causal loss
supervision mask to prevent cascading error noise.
"""

import json
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from spatial_pooling import StepSpanExtractor, MeanStepPooling, LastTokenStepPooling

MODEL_ID = "Qwen/Qwen3-0.6B"
DATA_PATH = "data/assignment3_data.json"

def extract_split(split_data: list, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                  device: str, max_steps: int, num_layers: int, hidden_dim: int, split_name: str = "split") -> dict:
    """
    Extracts multi layer hidden states and applies Causal Loss Masking.
    Dynamically adapts to model's num_layers (28) and hidden_dim (1024).
    """
    step_embeddings_mean = []
    step_embeddings_last = []
    step_targets = []
    step_masks = []
    gold_first_errors = []
    step_token_lengths = []

    mean_pooler = MeanStepPooling()
    last_pooler = LastTokenStepPooling()

    print(f"\nProcessing '{split_name}' split ({len(split_data)} examples, layers={num_layers}, dim={hidden_dim}, max_steps={max_steps})")

    for i, ex in enumerate(tqdm(split_data)):
        verdict = ex.get("verdict", "CORRECT")
        first_error = ex.get("first_error", None)
        
        gold_first_error = 0 if (first_error is None or verdict == "CORRECT") else int(first_error)

        messages = [
            {"role": "system", "content": "You solve arithmetic word problems accurately."},
            {"role": "user", "content": f"Question:\n{ex['question']}\n\nCandidate solution:\n{ex['candidate']}"}
        ]
        # Disable Qwen3 thinking tags in prompt template so offset mapping stays clean
        try:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        enc = tokenizer(full_prompt, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = torch.tensor([enc["input_ids"]], device=device)
        
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
            outputs = model(input_ids=input_ids, output_hidden_states=True,use_cache=False,)
            all_layers = torch.stack(outputs.hidden_states[1:], dim=0)[:, 0, :, :] # (L, Seq_Len, D)
            del outputs

        step_means = []
        step_lasts = []
        step_lens = []

        for span in extractor:
            step_slice = all_layers[:, span.tok_start:span.tok_end, :]
            step_means.append(mean_pooler(step_slice)) 
            step_lasts.append(last_pooler(step_slice))
            step_lens.append(span.num_tokens)
        del all_layers

        step_means = torch.stack(step_means, dim=1)  # (L, num_steps, D)
        step_lasts = torch.stack(step_lasts, dim=1)  # (L, num_steps, D)

        padded_mean = torch.zeros((num_layers, max_steps, hidden_dim), dtype=torch.float16)
        padded_last = torch.zeros((num_layers, max_steps, hidden_dim), dtype=torch.float16)

        padded_mean[:, :num_steps, :] = step_means.half().cpu()
        padded_last[:, :num_steps, :] = step_lasts.half().cpu()
        del step_means, step_lasts
        targets = torch.zeros(max_steps, dtype=torch.float32)
        masks = torch.zeros(max_steps, dtype=torch.float32)
        lens = torch.zeros(max_steps, dtype=torch.long)

        lens[:num_steps] = torch.tensor(step_lens, dtype=torch.long)

        # CAUSAL LOSS MASKING
        if gold_first_error == 0:
            masks[:num_steps] = 1.0
        else:
            k = gold_first_error
            masks[:k] = 1.0
            targets[k - 1] = 1.0

        step_embeddings_mean.append(padded_mean)
        step_embeddings_last.append(padded_last)
        step_targets.append(targets)
        step_masks.append(masks)
        gold_first_errors.append(gold_first_error)
        step_token_lengths.append(lens)

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

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Model Config Detected -> Layers: {num_layers}, Hidden Dim: {hidden_dim}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing data file at '{DATA_PATH}'.")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    max_steps = max(ex["num_steps"] for split in splits.values() for ex in split)
    print(f"Maximum reasoning steps across dataset: {max_steps}")

    os.makedirs("data", exist_ok=True)

    train_cache = extract_split(splits["train"], model, tokenizer, device, max_steps, num_layers, hidden_dim, "train")
    torch.save(train_cache, "data/train_cached.pt")
    print("Saved -> data/train_cached.pt")

    val_cache = extract_split(splits["validation"], model, tokenizer, device, max_steps, num_layers, hidden_dim, "validation")
    torch.save(val_cache, "data/val_cached.pt")
    print("Saved -> data/val_cached.pt")

    test_cache = extract_split(splits["test"], model, tokenizer, device, max_steps, num_layers, hidden_dim, "test")
    torch.save(test_cache, "data/test_cached.pt")
    print("Saved -> data/test_cached.pt")

    print(f"\nSUCCESS All {MODEL_ID} representations extracted and cached.")

if __name__ == "__main__":
    main()