"""
train_v9.py — LoRA fine-tune for the fan_center adapter.

Reads train_v9.jsonl produced by gen_v9.py. It does NOT generate data itself:
generation and training are separate so the data can be inspected before an hour of
CPU time is spent on it. Every previous failure was visible in the data.

Prompt schema (must match buildPrompt() in src/model.js):
    user   : ssb_rsrp_dBm=[...] (beam azimuths [...] deg)
    answer : {"fan_center": <float>, "action": "follow", "reason": "..."}

Note there is NO current_fan_center. That scalar was what the model copied.

Run:
    python gen_v9.py --n 1500      # first, and read the diagnostics
    python train_v7.py             # then this
"""

import json, os, random
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA       = "train_v9.jsonl"
OUT_DIR    = "beam-lora-v9"
EPOCHS     = 2
LR         = 2e-4
LORA_R     = 16
LORA_ALPHA = 32
MAX_LEN    = 256
SEED       = 0
random.seed(SEED); torch.manual_seed(SEED)

# ---------------------------------------------------------------- load data
if not os.path.exists(DATA):
    raise SystemExit(f"{DATA} not found. Run:  python gen_v9.py --n 1500")

rows = [json.loads(l) for l in open(DATA)]
print(f"loaded {len(rows)} samples from {DATA}")
print()
print("example the model will be trained on:")
print("  user  :", rows[0]["messages"][1]["content"])
print("  answer:", rows[0]["messages"][2]["content"])
print()
if "current_fan_center" in rows[0]["messages"][1]["content"]:
    raise SystemExit("ERROR: prompt still contains current_fan_center — regenerate the data")

# ---------------------------------------------------------------- model (CPU)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.to("cpu")

def generate(messages):
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=60, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

probe = rows[0]["messages"]
print("--- BEFORE (base model) ---")
print("target:", probe[2]["content"])
print("model :", generate(probe))
print()

# ---------------------------------------------------------------- LoRA
lora = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

def encode(r):
    text = tok.apply_chat_template(r["messages"], tokenize=False)
    return tok(text, truncation=True, max_length=MAX_LEN)

ds = Dataset.from_list([encode(r) for r in rows])
collator = DataCollatorForLanguageModeling(tok, mlm=False)

args = TrainingArguments(
    output_dir="trainer_tmp_v9",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    logging_steps=25,
    save_strategy="epoch",      # keep a checkpoint per epoch; a late crash is not fatal
    save_total_limit=2,
    report_to="none",
    use_cpu=True,
)

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
print(f"\ntraining on {len(rows)} samples, {EPOCHS} epochs, CPU...")
trainer.train()

model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"\nadapter saved to ./{OUT_DIR}")

# ---------------------------------------------------------------- checks
print("\n--- AFTER (trained) ---")
print("target:", probe[2]["content"])
print("model :", generate(probe))

print("\n--- COPY TEST: the failure mode this run exists to kill ---")
print("Same RSRP profile, different beam frames. If it reads the signal the answers")
print("should be the same absolute angle. If it copies, they will differ.")
for r in rows[:6]:
    user = r["messages"][1]["content"]
    a = json.loads(r["messages"][2]["content"])
    got = generate(r["messages"])
    print(f"  target fan {a['fan_center']:+7.1f}  tilt {a['tilt']:5.1f}  ->  {got[:78]}")
