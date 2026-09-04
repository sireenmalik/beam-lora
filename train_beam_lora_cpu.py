"""
train_beam_lora_cpu.py
----------------------
CPU-only LoRA fine-tune of Qwen2.5-0.5B-Instruct on crowd-following beam traces.
No GPU. No bitsandbytes. No Unsloth. Plain peft + transformers, the route we picked.

What it does, end to end, in one run:
  1. loads Qwen2.5-0.5B-Instruct on CPU
  2. reads your trace .json files from ./traces  (or makes synthetic ones if empty)
  3. turns each tick into a chat example: system+rules -> counts -> JSON answer
  4. attaches a LoRA adapter and trains it on CPU
  5. saves the adapter to ./beam-lora
  6. prints a before/after generation on one held-out tick so you can see it moved

Install once (CPU):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers peft datasets accelerate

Run:
  python train_beam_lora_cpu.py
"""

import os, json, glob, random
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ------------------------------------------------------------------ CONFIG
MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"   # smallest instruct Qwen, ~1 GB, Apache 2.0
TRACE_DIR   = "traces"                        # put your collected .json files here
OUT_DIR     = "beam-lora"                     # adapter is saved here
N_SYNTH     = 30                              # synthetic ticks if TRACE_DIR is empty
EPOCHS      = 10                              # small data needs several passes
LR          = 2e-4
LORA_R      = 16
LORA_ALPHA  = 32
MAX_LEN     = 512
SEED        = 0
random.seed(SEED); torch.manual_seed(SEED)

# Five beams, matching the deck. fan_center range -49..49, tilt 3..45.
BEAM_AZ = [-40, -20, 0, 20, 40]

SYSTEM = (
    "You are a Non-RT RIC rApp steering a fan of uplink beams to follow a moving crowd. "
    "You are given per-beam UE counts and the current beam config. "
    "Return ONLY one JSON object with keys: fan_center (-49..49), tilt (3..45), "
    "action (follow|widen|allocate), reason (short). No prose, no thinking, JSON only."
)

# ------------------------------------------------------------------ DATA
# Expected shape of each tick in your .json files (a file may hold one dict or a list):
#   {"counts": [c0,c1,c2,c3,c4],
#    "fan_center": <int deg>, "tilt": <int deg>,
#    "action": "follow", "reason": "leading the crowd left"}
# If your real traces use different keys, edit parse_tick() only. Nothing else changes.

def parse_tick(t):
    return {
        "counts": [int(x) for x in t["counts"]],
        "fan_center": int(round(t["fan_center"])),
        "tilt": int(round(t["tilt"])),
        "action": t.get("action", "follow"),
        "reason": t.get("reason", ""),
    }

def load_traces(folder):
    ticks = []
    for path in glob.glob(os.path.join(folder, "*.json")):
        with open(path) as f:
            obj = json.load(f)
        rows = obj if isinstance(obj, list) else [obj]
        for r in rows:
            try:
                ticks.append(parse_tick(r))
            except Exception as e:
                print(f"skip a row in {path}: {e}")
    return ticks

def make_synthetic(n):
    # A hotspot walks across the sector; counts peak near it; the label is the
    # count-weighted centroid, which is exactly the deck's own method. Good enough
    # to run the pipeline before your real traces land, and sane as a target.
    ticks = []
    for i in range(n):
        center = -40 + (80 * i / max(1, n - 1))          # hotspot azimuth sweeps L->R
        counts = [max(0, int(round(20 * pow(2.718, -((az - center) ** 2) / (2 * 12 ** 2)))))
                  for az in BEAM_AZ]
        total = sum(counts) or 1
        cen = sum(a * c for a, c in zip(BEAM_AZ, counts)) / total
        fan = max(-49, min(49, int(round(cen))))
        ticks.append({
            "counts": counts,
            "fan_center": fan,
            "tilt": 25,
            "action": "follow",
            "reason": "leading the crowd toward the count-weighted centroid",
        })
    return ticks

def to_messages(t):
    user = (f"per_beam_counts={t['counts']} (beam azimuths {BEAM_AZ} deg); "
            f"current fan_center={0}, tilt={20}")
    answer = json.dumps({
        "fan_center": t["fan_center"],
        "tilt": t["tilt"],
        "action": t["action"],
        "reason": t["reason"],
    })
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]

# ------------------------------------------------------------------ BUILD SET
ticks = load_traces(TRACE_DIR)
if ticks:
    print(f"loaded {len(ticks)} real ticks from ./{TRACE_DIR}")
else:
    print(f"no files in ./{TRACE_DIR}, using {N_SYNTH} synthetic ticks so the run works")
    ticks = make_synthetic(N_SYNTH)

held = ticks[-1]                 # keep the last one out for the before/after test
train_ticks = ticks[:-1] if len(ticks) > 1 else ticks

# ------------------------------------------------------------------ MODEL (CPU)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.to("cpu")

def generate(sample_tick):
    msgs = to_messages(sample_tick)[:-1]          # drop the answer, ask the model
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80, do_sample=False)  # temp 0
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("\n--- BEFORE (base model) ---")
print("target:", json.dumps({k: held[k] for k in ('fan_center','tilt','action')}))
print("model :", generate(held))

# ------------------------------------------------------------------ LoRA
lora = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# tokenize the training chat examples
def encode(t):
    text = tok.apply_chat_template(to_messages(t), tokenize=False)
    enc = tok(text, truncation=True, max_length=MAX_LEN)
    return enc

ds = Dataset.from_list([encode(t) for t in train_ticks])
collator = DataCollatorForLanguageModeling(tok, mlm=False)  # causal LM, sets labels

args = TrainingArguments(
    output_dir="trainer_tmp",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    use_cpu=True,            # older transformers: replace with no_cuda=True
)

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
trainer.train()

# ------------------------------------------------------------------ SAVE + TEST
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"\nadapter saved to ./{OUT_DIR}")

print("\n--- AFTER (LoRA) ---")
print("target:", json.dumps({k: held[k] for k in ('fan_center','tilt','action')}))
print("model :", generate(held))
print("\nDone. To use it, load the base model then load_adapter('./beam-lora'),")
print("or export to GGUF later for Ollama. The formatter tool stays on as the safety net.")
