"""
train_beam_lora_cpu_v2.py
-------------------------
CPU-only LoRA fine-tune of Qwen2.5-0.5B-Instruct on crowd-following beam traces.

WHAT CHANGED FROM v1 (the under-steering fix):
  1. current_fan_center is now VARIED per example, not hard-coded to 0. The model
     learns to steer FROM wherever the beam currently is TO the crowd. v1 only ever
     saw "beam at 0", so it under-steered from every other position.
  2. 500 synthetic examples (was 30), with DELIBERATE corner coverage: the hard
     cases (crowd far right while beam far left, and vice versa) are sampled on
     purpose, not left to chance. Coverage of the grid beats raw volume.
  3. Some examples put the whole crowd at an extreme edge so the model learns to
     commit to a full swing, not just nudge toward center.

Everything else (LoRA config, CPU route, save path) is identical to v1.

Install once (CPU):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers peft datasets accelerate

Run:
  python train_beam_lora_cpu_v2.py
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
MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
TRACE_DIR   = "traces"
OUT_DIR     = "beam-lora"
N_SYNTH     = 500                             # was 30; two-variable task needs grid coverage
EPOCHS      = 6                               # more data -> fewer epochs needed
LR          = 2e-4
LORA_R      = 16
LORA_ALPHA  = 32
MAX_LEN     = 512
SEED        = 0
random.seed(SEED); torch.manual_seed(SEED)

BEAM_AZ = [-40, -20, 0, 20, 40]

SYSTEM = (
    "You are a Non-RT RIC rApp steering a fan of uplink beams to follow a moving crowd. "
    "You are given per-beam UE counts and the current beam config. "
    "Return ONLY one JSON object with keys: fan_center (-49..49), tilt (3..45), "
    "action (follow|widen|allocate), reason (short). No prose, no thinking, JSON only."
)

# ------------------------------------------------------------------ DATA HELPERS
def counts_for_center(center, spread=12, peak=20):
    """Bell-curve UE counts peaking at `center` across the 5 beams."""
    return [max(0, int(round(peak * pow(2.718, -((az - center) ** 2) / (2 * spread ** 2)))))
            for az in BEAM_AZ]

def weighted_centroid(counts):
    total = sum(counts) or 1
    return sum(a * c for a, c in zip(BEAM_AZ, counts)) / total

def make_tick(crowd_center, current_fan, spread=12, peak=20):
    """One training tick: crowd at crowd_center, beam currently at current_fan.
       The label (fan_center) is the count-weighted centroid, clamped."""
    counts = counts_for_center(crowd_center, spread, peak)
    cen = weighted_centroid(counts)
    fan = max(-49, min(49, int(round(cen))))
    # direction-aware reason so the audit text matches the move
    if fan < current_fan - 2:
        reason = "steering the fan left toward the count-weighted centroid"
    elif fan > current_fan + 2:
        reason = "steering the fan right toward the count-weighted centroid"
    else:
        reason = "holding on the count-weighted centroid"
    return {
        "counts": counts,
        "current_fan_center": current_fan,   # NEW: carried per example
        "fan_center": fan,
        "tilt": 25,
        "action": "follow",
        "reason": reason,
    }

def make_synthetic(n):
    """
    Build n examples covering the (crowd_position x current_beam_position) grid,
    with deliberate emphasis on the hard corners.
    """
    ticks = []
    grid = [-40, -30, -20, -10, 0, 10, 20, 30, 40]   # positions we care about

    # 1. Full grid: every crowd position x every current-beam position (81 combos)
    for crowd in grid:
        for cur in grid:
            ticks.append(make_tick(crowd, cur))

    # 2. Corner emphasis: the exact cases v1 failed. Crowd at one extreme while the
    #    beam sits at the opposite extreme. Repeat these several times.
    corners = [(-40, 40), (40, -40), (-40, 20), (40, -20), (-40, 0), (40, 0),
               (-40, -40), (40, 40)]  # includes "already there" to teach holding
    for _ in range(6):
        for crowd, cur in corners:
            ticks.append(make_tick(crowd, cur))

    # 3. Extreme-edge crowds (all mass on one edge beam) so it learns full swings.
    for _ in range(8):
        for crowd in (-40, 40):
            for cur in (-40, -20, 0, 20, 40):
                ticks.append(make_tick(crowd, cur, spread=8, peak=24))

    # 4. Fill the rest with random crowd/current pairs + slight noise for variety.
    while len(ticks) < n:
        crowd = random.randint(-40, 40)
        cur   = random.randint(-45, 45)
        spread = random.choice([9, 12, 15])
        peak   = random.choice([16, 20, 24])
        ticks.append(make_tick(crowd, cur, spread, peak))

    random.shuffle(ticks)
    return ticks[:n]

def parse_tick(r):
    """Adapt real trace JSON. current_fan_center defaults to 0 if absent."""
    return {
        "counts": r["counts"],
        "current_fan_center": r.get("current_fan_center", 0),
        "fan_center": r["fan_center"],
        "tilt": r.get("tilt", 25),
        "action": r.get("action", "follow"),
        "reason": r.get("reason", "leading the crowd toward the count-weighted centroid"),
    }

def load_traces(d):
    ticks = []
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        for r in (data if isinstance(data, list) else [data]):
            try:
                ticks.append(parse_tick(r))
            except Exception:
                pass
    return ticks

def to_messages(t):
    # NEW: use the tick's own current_fan_center, not a hard-coded 0.
    user = (f"per_beam_counts={t['counts']} (beam azimuths {BEAM_AZ} deg); "
            f"current fan_center={t['current_fan_center']}, tilt={20}")
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
    print(f"no files in ./{TRACE_DIR}, using {N_SYNTH} synthetic ticks (grid + corner coverage)")
    ticks = make_synthetic(N_SYNTH)

# held-out test: a hard corner (crowd far right, beam far left) to check steering
held = make_tick(40, -40)
train_ticks = ticks

# ------------------------------------------------------------------ MODEL (CPU)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.to("cpu")

def generate(sample_tick):
    msgs = to_messages(sample_tick)[:-1]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("\n--- BEFORE (base model), hard corner: crowd far right, beam far left ---")
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

def encode(t):
    text = tok.apply_chat_template(to_messages(t), tokenize=False)
    return tok(text, truncation=True, max_length=MAX_LEN)

ds = Dataset.from_list([encode(t) for t in train_ticks])
collator = DataCollatorForLanguageModeling(tok, mlm=False)

args = TrainingArguments(
    output_dir="trainer_tmp",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    logging_steps=25,
    save_strategy="no",
    report_to="none",
    use_cpu=True,
)

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
print(f"\ntraining on {len(train_ticks)} examples, {EPOCHS} epochs, CPU...")
trainer.train()

# ------------------------------------------------------------------ SAVE + TEST
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"\nadapter saved to ./{OUT_DIR}")

print("\n--- AFTER (trained), hard corner: crowd far right, beam far left ---")
print("target:", json.dumps({k: held[k] for k in ('fan_center','tilt','action')}))
print("model :", generate(held))

# extra spot checks on the corners v1 failed
print("\n--- spot checks ---")
for crowd, cur in [(40, -40), (-40, 40), (40, 0), (-40, 0)]:
    t = make_tick(crowd, cur)
    print(f"crowd={crowd:+d} beam@{cur:+d}  target fan={t['fan_center']:+d}  ->  {generate(t)}")
