"""
train_beam_lora_cpu_v3.py
-------------------------
CPU-only LoRA fine-tune of Qwen2.5-0.5B-Instruct for the crowd-following beam rApp.

WHAT CHANGED FROM v2 (the RSRP fix):
  v2 trained on per-beam UE COUNTS, e.g. [0, 4, 51, 2, 0].
  The rApp no longer sends counts. A real gNodeB cannot report per-beam UE counts —
  3GPP defines no such counter over O1. What it does report is SS-RSRP per SSB beam.
  The loop now sends profiles like [-63, -63, -62, -43, -33] dBm.

  The v2 adapter, fed RSRP, saturates at +/-29 because those numbers are nothing
  like its training distribution. This script retrains on the real input format.

  The RSRP physics below are a direct port of geometry.js so that the training input
  is byte-identical in shape to what the model sees at inference. That match is the
  whole point: the previous failure was a train/inference format mismatch.

Everything else (LoRA config, CPU route, save path, corner coverage) is as v2.

Install once (CPU):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers peft datasets accelerate

Run:
  python train_beam_lora_cpu_v3.py
"""

import os, json, glob, random, math
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
OUT_DIR     = "beam-lora-v6"
N_SYNTH     = 1200
EPOCHS      = 2
LR          = 2e-4
LORA_R      = 16
LORA_ALPHA  = 32
MAX_LEN     = 256
SEED        = 0
random.seed(SEED); torch.manual_seed(SEED)

# ------------------------------------------------------------------ RF MODEL
# Direct port of src/geometry.js. Keep these in sync with the demo.
TOWER_H       = 25.0     # antenna height, metres
N_BEAMS       = 5
FAN_SPAN      = 30.0     # beams spread across fanCenter +/- FAN_SPAN
FC_GHZ        = 3.5
P_TX_DBM      = 18.0
G_MAX_DBI     = 15.0
HPBW_DEG      = 20.0     # half-power beamwidth (3GPP parabolic pattern)
FRONT_BACK_DB = 30.0
SF_SIGMA_DB   = 4.0
RSRP_MIN      = -156
RSRP_MAX      = -31

def fan_azimuths(fan_center):
    step = (2 * FAN_SPAN) / (N_BEAMS - 1)
    return [fan_center - FAN_SPAN + i * step for i in range(N_BEAMS)]

def path_loss_38901(d):
    """3GPP TR 38.901 UMi-Street-Canyon LOS."""
    dd = max(1.0, d)
    return 32.4 + 21 * math.log10(dd) + 20 * math.log10(FC_GHZ)

def beam_gain_db(az_offset):
    """3GPP parabolic element pattern."""
    off = abs(az_offset)
    atten = min(12 * (off / HPBW_DEG) ** 2, FRONT_BACK_DB)
    return G_MAX_DBI - atten

def quantize_rsrp(dbm):
    return int(max(RSRP_MIN, min(RSRP_MAX, round(dbm))))

def rsrp_per_beam(ue_azimuths, ue_ranges, fan_center, shadows=None):
    """
    Port of rsrpPerBeam(). Every beam hears every UE, at a strength set by how far
    off boresight it is. Reports TOTAL received power per beam.

    `shadows` is the per-UE shadow fade. It is passed IN rather than redrawn here,
    because shadow fading is a property of where the UE is standing (buildings block
    the path), not a fresh dice roll every tick. Redrawing it per call modelled white
    noise and made a stationary crowd appear to swing several degrees per tick.
    """
    azs = fan_azimuths(fan_center)
    lin = [0.0] * N_BEAMS
    if shadows is None:
        shadows = [random.gauss(0, SF_SIGMA_DB) for _ in ue_azimuths]
    for az, rng, sf in zip(ue_azimuths, ue_ranges, shadows):
        d3d = math.hypot(rng, TOWER_H)
        pl = path_loss_38901(d3d)
        for b in range(N_BEAMS):
            rsrp = P_TX_DBM - pl + beam_gain_db(az - azs[b]) + sf
            lin[b] += 10 ** (rsrp / 10)
    return [quantize_rsrp(10 * math.log10(v)) if v > 0 else RSRP_MIN for v in lin]

def rsrp_centroid(rsrp, fan_center):
    """Port of rsrpCentroid(). Weight on CONTRAST above the weakest beam."""
    azs = fan_azimuths(fan_center)
    served = [r for r in rsrp if r > RSRP_MIN]
    if not served:
        return fan_center
    floor = min(served)
    lin = [0.0 if r <= RSRP_MIN else 10 ** ((r - floor) / 10) for r in rsrp]
    tot = sum(lin)
    if tot <= 0:
        return fan_center
    return sum(w * a for w, a in zip(lin, azs)) / tot

SYSTEM = (
    "You are a Non-RT RIC rApp steering a grid of uplink beams toward the load in a cell. "
    "You are given SS-RSRP per SSB beam in dBm and the current beam config. "
    "Return ONLY one JSON object with keys: fan_center (-49..49), tilt (3..45), "
    "action (follow|widen|allocate), reason (short). No prose, no thinking, JSON only."
)

# ------------------------------------------------------------------ DATA HELPERS
def make_tick(crowd_center, current_fan, spread_deg=6.0, rng_m=100.0, n_ue=60):
    """
    One training tick, generated the SAME way the live loop produces its input:

      1. shadow fading is fixed per UE (a property of where it stands)
      2. the crowd is observed TWICE, one tick apart, drifting slightly between
      3. the two RSRP reports are averaged in linear power  <- the live pre-loop does this
      4. the label is the centroid of that averaged profile <- what the arithmetic gets

    Point 3 matters: the model is fed an averaged profile at inference, so it must be
    trained on an averaged profile. Training on a single noisy report would be a
    different input distribution again.
    """
    az = [random.gauss(crowd_center, spread_deg) for _ in range(n_ue)]
    rg = [max(20.0, random.gauss(rng_m, rng_m * 0.12)) for _ in range(n_ue)]
    shadows = [random.gauss(0, SF_SIGMA_DB) for _ in range(n_ue)]   # fixed per UE

    # previous report: the crowd was a fraction of a tick behind
    back = (crowd_center - current_fan) * 0.15
    az_prev = [a - back for a in az]
    r_prev = rsrp_per_beam(az_prev, rg, current_fan, shadows)
    r_now  = rsrp_per_beam(az,      rg, current_fan, shadows)

    # average the two reports in linear power, same as the live pre-loop
    rsrp = []
    for b in range(N_BEAMS):
        lin = (10 ** (r_prev[b] / 10) + 10 ** (r_now[b] / 10)) / 2
        rsrp.append(quantize_rsrp(10 * math.log10(lin)) if lin > 0 else RSRP_MIN)

    # LABEL = what the arithmetic gets from the averaged profile. Nothing else.
    fan = round(max(-49, min(49, rsrp_centroid(rsrp, current_fan))), 1)

    # The live loop settles: once the beam is on a stationary crowd it stops moving,
    # because the fade is fixed and the same profile arrives every tick. Reproduce that
    # here. Without it every "crowd is still" example carries a fresh random offset and
    # teaches the beam to shuffle when nothing has happened.
    if abs(crowd_center - current_fan) < 0.25:
        fan = round(current_fan, 1)

    tilt = int(round(max(3, min(45, math.degrees(math.atan2(TOWER_H, sum(rg) / len(rg)))))))

    if fan < current_fan - 0.5:
        reason = "steering left toward the RSRP peak"
    elif fan > current_fan + 0.5:
        reason = "steering right toward the RSRP peak"
    else:
        reason = "holding on the RSRP peak"

    return {
        "rsrp": rsrp,
        "beam_az": [round(a, 1) for a in fan_azimuths(current_fan)],
        "current_fan_center": round(current_fan, 1),
        "fan_center": fan,
        "tilt": tilt,
        "action": "follow",
        "reason": reason,
    }

def make_synthetic(n):
    """
    v5: trained on the REAL operating envelope, measured from the running demo.

    The beam is placed on the crowd at startup, so it is never parked far away waiting
    to acquire. From then on:
        Auto Drift    crowd centroid moves ~0.9 deg per tick
        Crowd Linear  clicking a target makes the crowd WALK there at ~3 deg per tick max
        Chaos         crowd disperses but the centroid barely moves
    Measured largest beam move over 25 ticks: 4.4 deg.

    So the entire task is: "you are on the crowd, it drifted a degree or two, follow it."
    v3 and v4 wasted 10-30% of their examples on 25-65 degree swings that CANNOT occur.
    Those examples are what taught the model to emit big, bucketed values that the
    guardrails then had to catch. The fences and the training were fighting each other,
    and both were mine.

    v5 trains only inside the envelope, at fine granularity.
    """
    ticks = []

    # Beam positions across the sector — the beam can sit anywhere, it is the MOVE that
    # is small. Fine steps so the model sees many distinct current positions.
    positions = [x * 2.5 for x in range(-16, 17)]        # -40 .. +40 in 2.5 deg steps

    # Moves within the measured envelope, at 0.5 deg resolution, both directions,
    # including zero (the crowd stopped; hold).
    deltas = [-4.5, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
              0.0,
              0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5]

    for _ in range(2):
        for cur in positions:
            for d in deltas:
                crowd = max(-44, min(44, cur + d))
                ticks.append(make_tick(crowd, cur))

    # A little variety in crowd tightness and range, still inside the envelope.
    while len(ticks) < n:
        cur = random.uniform(-42, 42)
        d = random.uniform(-4.5, 4.5)
        ticks.append(make_tick(
            max(-44, min(44, cur + d)), cur,
            spread_deg=random.choice([4.0, 6.0, 9.0]),
            rng_m=random.choice([70.0, 100.0, 130.0]),
        ))

    random.shuffle(ticks)
    return ticks[:n]

def parse_tick(r):
    """Adapt a real trace JSON produced by the rApp."""
    cur = r.get("current_fan_center", 0)
    return {
        "rsrp": r["rsrp"],
        "beam_az": r.get("beam_az", [round(a, 1) for a in fan_azimuths(cur)]),
        "current_fan_center": cur,
        "fan_center": r["fan_center"],
        "tilt": r.get("tilt", 25),
        "action": r.get("action", "follow"),
        "reason": r.get("reason", "steering toward the RSRP peak"),
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
    """
    MUST match the prompt the rApp sends at inference (src/model.js buildPrompt).
    Field names and ordering are deliberately the same.
    """
    user = (f"ssb_rsrp_dBm={t['rsrp']} (beam azimuths {t['beam_az']} deg); "
            f"current fan_center={t['current_fan_center']}, tilt={t['tilt']}")
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
    print(f"no files in ./{TRACE_DIR}, using {N_SYNTH} synthetic ticks (RSRP, grid + corners)")
    ticks = make_synthetic(N_SYNTH)

held = make_tick(40, -40)          # hard corner: crowd far right, beam far left
train_ticks = ticks

print("\nexample training input (this must look like what the rApp sends):")
print(" ", to_messages(train_ticks[0])[1]["content"])
print("  label:", to_messages(train_ticks[0])[2]["content"])

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
print("target:", json.dumps({k: held[k] for k in ('fan_center', 'tilt', 'action')}))
print("model :", generate(held))

# ------------------------------------------------------------------ LoRA
lora = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
print("target:", json.dumps({k: held[k] for k in ('fan_center', 'tilt', 'action')}))
print("model :", generate(held))

print("\n--- spot checks: the REAL envelope (crowd drifts 0.5-4 deg from the beam) ---")
for cur, d in [(0, 1.0), (0, -1.0), (0, 2.5), (0, -2.5),
               (20, 1.5), (-20, -1.5), (35, 2.0), (-35, -2.0), (0, 0.0)]:
    t = make_tick(cur + d, cur)
    print(f"beam@{cur:+6.1f}  crowd drifted {d:+4.1f}  ->  target fan={t['fan_center']:+3d}  ->  {generate(t)}")
