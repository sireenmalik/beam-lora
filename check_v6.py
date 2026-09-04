"""
check_v6.py — evaluate the trained adapter without retraining.

Run:
  cd C:\\beam-lora
  python check_v6.py
"""
import json, math, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER    = "beam-lora-v6"
random.seed(3)

# ---- RF model (same as the training script and the live loop) ----
TOWER_H, N_BEAMS, FAN_SPAN = 25.0, 5, 30.0
FC_GHZ, P_TX_DBM, G_MAX_DBI = 3.5, 18.0, 15.0
HPBW_DEG, FRONT_BACK_DB, SF_SIGMA_DB = 20.0, 30.0, 4.0
RSRP_MIN, RSRP_MAX = -156, -31

def fan_azimuths(fc):
    step = (2 * FAN_SPAN) / (N_BEAMS - 1)
    return [fc - FAN_SPAN + i * step for i in range(N_BEAMS)]

def path_loss(d):
    return 32.4 + 21 * math.log10(max(1.0, d)) + 20 * math.log10(FC_GHZ)

def gain(off):
    return G_MAX_DBI - min(12 * (abs(off) / HPBW_DEG) ** 2, FRONT_BACK_DB)

def q(dbm):
    return int(max(RSRP_MIN, min(RSRP_MAX, round(dbm))))

def rsrp_per_beam(azs_ue, rgs, fc, shadows):
    azs = fan_azimuths(fc)
    lin = [0.0] * N_BEAMS
    for a, r, sf in zip(azs_ue, rgs, shadows):
        pl = path_loss(math.hypot(r, TOWER_H))
        for b in range(N_BEAMS):
            lin[b] += 10 ** ((P_TX_DBM - pl + gain(a - azs[b]) + sf) / 10)
    return [q(10 * math.log10(v)) if v > 0 else RSRP_MIN for v in lin]

def rsrp_centroid(rsrp, fc):
    azs = fan_azimuths(fc)
    served = [r for r in rsrp if r > RSRP_MIN]
    if not served: return fc
    floor = min(served)
    lin = [0.0 if r <= RSRP_MIN else 10 ** ((r - floor) / 10) for r in rsrp]
    tot = sum(lin)
    return fc if tot <= 0 else sum(w * a for w, a in zip(lin, azs)) / tot

def make_tick(crowd, cur, spread=6.0, rng=100.0, n=60):
    az = [random.gauss(crowd, spread) for _ in range(n)]
    rg = [max(20.0, random.gauss(rng, rng * 0.12)) for _ in range(n)]
    sh = [random.gauss(0, SF_SIGMA_DB) for _ in range(n)]
    back = (crowd - cur) * 0.15
    p = rsrp_per_beam([a - back for a in az], rg, cur, sh)
    c = rsrp_per_beam(az, rg, cur, sh)
    rsrp = [q(10 * math.log10((10 ** (p[b] / 10) + 10 ** (c[b] / 10)) / 2)) for b in range(N_BEAMS)]
    fan = round(max(-49, min(49, rsrp_centroid(rsrp, cur))), 1)
    if abs(crowd - cur) < 0.25:
        fan = round(cur, 1)
    tilt = int(round(max(3, min(45, math.degrees(math.atan2(TOWER_H, sum(rg) / len(rg)))))))
    return {"rsrp": rsrp, "beam_az": [round(a, 1) for a in fan_azimuths(cur)],
            "current_fan_center": round(cur, 1), "fan_center": fan, "tilt": tilt}

SYSTEM = ("You are a Non-RT RIC rApp steering a grid of uplink beams toward the load in a cell. "
          "You are given SS-RSRP per SSB beam in dBm and the current beam config. "
          "Return ONLY one JSON object with keys: fan_center (-49..49), tilt (3..45), "
          "action (follow|widen|allocate), reason (short). No prose, no thinking, JSON only.")

print("loading model + adapter...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def ask(t):
    user = (f"ssb_rsrp_dBm={t['rsrp']} (beam azimuths {t['beam_az']} deg); "
            f"current fan_center={t['current_fan_center']}, tilt={t['tilt']}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=60, do_sample=False)
    txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        m = txt[txt.index("{"):txt.rindex("}") + 1]
        return float(json.loads(m)["fan_center"]), txt
    except Exception:
        return None, txt

print()
print("THE REAL ENVELOPE: beam is on the crowd, crowd drifts a degree or two")
print("=" * 78)
print(f"{'beam at':>8} {'drift':>7} {'target':>8} {'model':>8} {'error':>8}   verdict")
print("-" * 78)

cases = [(0, 0.0), (0, 1.0), (0, -1.0), (0, 2.0), (0, -2.0), (0, 3.0), (0, -3.0),
         (20, 1.5), (20, -1.5), (-20, 1.5), (-20, -1.5),
         (35, 2.0), (-35, -2.0), (10, 0.5), (-10, -0.5)]
errs, dirs = [], []
for cur, d in cases:
    t = make_tick(cur + d, cur)
    got, raw = ask(t)
    if got is None:
        print(f"{cur:8.1f} {d:+7.1f} {t['fan_center']:8.1f}   PARSE FAIL: {raw[:40]}")
        continue
    err = got - t['fan_center']
    errs.append(abs(err))
    want = t['fan_center'] - t['current_fan_center']
    move = got - t['current_fan_center']
    ok = (want == 0 and abs(move) < 1) or (want != 0 and move * want > 0)
    dirs.append(ok)
    print(f"{cur:8.1f} {d:+7.1f} {t['fan_center']:8.1f} {got:8.1f} {err:+8.1f}   {'OK' if ok else 'WRONG WAY'}")

print("-" * 78)
if errs:
    print(f"  mean absolute error : {sum(errs)/len(errs):.2f} deg")
    print(f"  correct direction   : {sum(dirs)}/{len(dirs)}")
    print(f"  distinct answers    : {len(set(round(e,1) for e in errs))}  (bucketing check)")
