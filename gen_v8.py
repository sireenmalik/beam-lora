"""
gen_v8.py — training data for the fan_center adapter.

DESIGN RULES (each one is a lesson from a previous failure):

  1. The physics is a LINE-FOR-LINE PORT of src/geometry.js, not a re-implementation.
     Every re-write so far diverged: wrong beamwidth (65 vs 20 deg), wrong tx power
     (43 vs 18 dBm), a stray -1.0 in the centroid, no path loss, single-point crowds.
     Train on one distribution and run on another and the model fails in the loop
     while passing every offline check. The constants below are copied from the demo.

  2. `current_fan_center` is NOT in the prompt.
     The weighted centroid is fully determined by the RSRP profile and the beam
     azimuths; the current beam position does not appear in the equation. Supplying it
     gave the model a scalar to copy, and it did exactly that - it echoed the input and
     the beam sat frozen while the crowd walked away. The absolute frame is carried by
     the beam azimuths, which remain in the prompt.

  3. Beam-to-crowd offsets are UNIFORM over +/-25 deg.
     Not because the demo does that (it runs at 0-4 deg) but because a training set
     where the answer is always near the input makes copying the winning strategy.
     Uniform offsets destroy that shortcut.

  4. +/-25 deg is MEASURED, not chosen.
     Beyond ~25 deg the crowd leaves the beam fan, the peak pins to the edge beam and
     the centroid saturates: many different crowd positions map to the same label.
     That many-to-one region is what produced the bucketed -10/-20/-33 outputs.

  5. The label is EXACTLY what rsrpCentroid() computes at runtime.

  KNOWN LIMIT, stated rather than hidden:
     Beyond roughly 45 deg of offset the crowd falls outside the beam fan entirely,
     every beam reads the same floor value, and rsrpCentroid() returns the fan centre.
     The label then EQUALS the beam position, so copying scores perfectly there. Such
     cases are excluded from training by THETA_MAX, which is correct - the profile
     carries no direction, so there is nothing to learn. But it does mean the model is
     untrained in that regime and will probably copy if it ever sees it. The demo never
     goes there (it runs at 0-4 deg of offset and starts locked on), so this is a
     documented boundary, not a live risk.
     Not the simulator's private knowledge of where the crowd stands. The model sees
     only the RSRP profile; asking it for a number that profile cannot yield is asking
     it to learn noise.

Usage:
    python gen_v8.py                 # writes train_v8.jsonl + prints diagnostics
    python gen_v8.py --n 2000        # more samples
"""

import argparse
import json
import math
import random
from collections import Counter

# ---------------------------------------------------------------------------
# CONSTANTS — copied verbatim from src/geometry.js. Do not "improve" these.
# ---------------------------------------------------------------------------
TOWER_H        = 25.0     # antenna height, metres
N_BEAMS        = 5        # grid-of-beams
FAN_SPAN       = 30.0     # fanAzimuths(fc, span=30)
FC_GHZ         = 3.5      # mid-band carrier
P_TX_DBM       = 18.0     # per-SSB transmit power
G_MAX_DBI      = 15.0     # peak beam gain at boresight
HPBW_DEG       = 20.0     # half-power beamwidth per beam
FRONT_BACK_DB  = 30.0     # side-lobe floor
SF_SIGMA_DB    = 4.0      # TR 38.901 UMi-LOS shadow fading sigma
RSRP_MIN       = -156     # TS 38.133 reportable floor
RSRP_MAX       = -31      # TS 38.133 reportable ceiling

THETA_MAX      = 25.0     # measured usable offset (see rule 4)
# The fan spans fan_center +/- 30 deg, so bounding the CENTRE at 45 would put the edge
# beam at 75 deg — outside any real sector. Bound the fan so its outermost beam stays
# within a 60 deg sector, i.e. |fan_center| + FAN_SPAN <= 60.
FAN_LIMIT      = 30.0     # max |fan_center|; edge beams then reach +/-60 deg
CROWD_LIMIT    = 40.0     # crowd can sit slightly outside the fan centre range

SYSTEM = (
    "You are a Non-RT RIC rApp steering a grid of uplink beams toward the load in a cell. "
    "You are given SS-RSRP per SSB beam in dBm and the azimuth each beam points at. "
    "Return ONLY one JSON object with keys: fan_center (-49..49), action "
    "(follow|widen|allocate), reason (short). No prose, no thinking, JSON only."
)

# ---------------------------------------------------------------------------
# PHYSICS — port of geometry.js
# ---------------------------------------------------------------------------
def fan_azimuths(fan_center):
    """Port of fanAzimuths(fc, span=30)."""
    step = (2 * FAN_SPAN) / (N_BEAMS - 1)
    return [fan_center - FAN_SPAN + i * step for i in range(N_BEAMS)]


def path_loss_38901(d):
    """3GPP TR 38.901 UMi-Street-Canyon LOS. d in metres, fc in GHz."""
    dd = max(1.0, d)
    return 32.4 + 21 * math.log10(dd) + 20 * math.log10(FC_GHZ)


def beam_gain_db(az_offset):
    """3GPP parabolic element pattern, capped at the side-lobe floor."""
    off = abs(az_offset)
    atten = min(12.0 * (off / HPBW_DEG) ** 2, FRONT_BACK_DB)
    return G_MAX_DBI - atten


def quantize_rsrp(dbm):
    """TS 38.133 integer dBm, clamped to the reportable range."""
    return int(max(RSRP_MIN, min(RSRP_MAX, round(dbm))))


def rsrp_per_beam(ue_az, ue_rng, ue_shadow, fan_center):
    """
    Port of rsrpPerBeam().

    EVERY beam hears EVERY UE, at a strength set by how far off boresight it is.
    Real grid-of-beams patterns overlap; a beam does not go silent because a user is
    better served elsewhere. Reports TOTAL received power per beam.

    Shadow fading is passed in, not drawn here: it is a property of where the UE is
    standing (buildings block the path), not a fresh dice roll per call.
    """
    azs = fan_azimuths(fan_center)
    lin = [0.0] * N_BEAMS
    for az, rng, sf in zip(ue_az, ue_rng, ue_shadow):
        d3d = math.hypot(rng, TOWER_H)
        pl = path_loss_38901(d3d)
        for b in range(N_BEAMS):
            rsrp = P_TX_DBM - pl + beam_gain_db(az - azs[b]) + sf
            lin[b] += 10 ** (rsrp / 10)
    # MEAN per UE, not the sum.
    #
    # RSRP is defined as the power ONE device receives; the standard counter is a
    # distribution of individual readings. Summing N users adds 10*log10(N) of gain that
    # was never in the channel - 18 dB at N=60, 30 dB at N=1000 - which pushes the value
    # past the TS 38.133 ceiling of -31 dBm. Past that the reading stops responding to
    # distance entirely. The mean never saturates and crowd size cancels.
    n = max(1, len(ue_az))
    return [quantize_rsrp(10 * math.log10(v / n)) if v > 0 else RSRP_MIN for v in lin]


def rsrp_centroid(rsrp, fan_center):
    """
    Port of rsrpCentroid(). THE LABEL FUNCTION.

    Weight on CONTRAST above the weakest served beam, not on absolute power:
    overlapping beams all hear the crowd, so absolute powers sit within a few dB and
    converting them straight to linear gives near-equal weights, collapsing the
    centroid toward boresight. Subtracting the floor first restores the contrast.

    No stray offsets. This must equal the runtime function exactly.
    """
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


# ---------------------------------------------------------------------------
# SAMPLE GENERATION
# ---------------------------------------------------------------------------
def make_sample(rng, n_ue=60):
    """
    One training sample.

    The crowd is placed first, then the beam is placed at a UNIFORM offset from it.
    That ordering matters: it makes (target - beam) uniform by construction, so no
    residual "usual correction size" exists for the model to exploit.
    """
    # crowd: azimuth anywhere in the sector, realistic spread and range
    crowd_az = rng.uniform(-CROWD_LIMIT, CROWD_LIMIT)
    spread   = rng.choice([4.0, 6.0, 9.0])
    crowd_r  = rng.choice([70.0, 100.0, 130.0])

    # beam: uniform offset from the crowd, kept inside the sector
    offset = rng.uniform(-THETA_MAX, THETA_MAX)
    fan_center = max(-FAN_LIMIT, min(FAN_LIMIT, crowd_az - offset))

    ue_az     = [rng.gauss(crowd_az, spread) for _ in range(n_ue)]
    ue_rng    = [max(20.0, rng.gauss(crowd_r, crowd_r * 0.12)) for _ in range(n_ue)]
    ue_shadow = [rng.gauss(0.0, SF_SIGMA_DB) for _ in range(n_ue)]   # fixed per UE

    rsrp = rsrp_per_beam(ue_az, ue_rng, ue_shadow, fan_center)
    label = round(max(-49.0, min(49.0, rsrp_centroid(rsrp, fan_center))), 1)

    azs = [round(a, 1) for a in fan_azimuths(fan_center)]
    if label < fan_center - 0.5:
        reason = "steering left toward the RSRP peak"
    elif label > fan_center + 0.5:
        reason = "steering right toward the RSRP peak"
    else:
        reason = "holding on the RSRP peak"

    return {
        "rsrp": rsrp,
        "beam_az": azs,
        "fan_center_now": round(fan_center, 1),   # bookkeeping only, NOT in the prompt
        "crowd_az": round(crowd_az, 1),           # bookkeeping only
        "label": label,
        "reason": reason,
    }


def to_messages(s):
    """
    The prompt. MUST match buildPrompt() in src/model.js exactly.

    Note what is absent: current_fan_center. The beam azimuths carry the frame.
    """
    user = f"ssb_rsrp_dBm={s['rsrp']} (beam azimuths {s['beam_az']} deg)"
    answer = json.dumps({
        "fan_center": s["label"],
        "action": "follow",
        "reason": s["reason"],
    })
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]


# ---------------------------------------------------------------------------
# DIAGNOSTICS — the checks that would have caught the previous failures
# ---------------------------------------------------------------------------
def diagnostics(samples):
    print()
    print("=" * 74)
    print("DIAGNOSTICS")
    print("=" * 74)

    offs = [s["label"] - s["fan_center_now"] for s in samples]
    n = len(offs)

    # 1. uniformity of (label - beam). A peak near zero means copying still pays.
    print()
    print("1. IS (label - beam) UNIFORM?  A spike at 0 means copying still works.")
    bins = Counter()
    for o in offs:
        k = int(math.floor(o / 5.0)) * 5
        bins[k] += 1
    for k in sorted(bins):
        bar = "#" * int(60 * bins[k] / max(bins.values()))
        print(f"   {k:+4d}..{k+5:+4d}  {bins[k]:5d}  {bar}")
    near_zero = sum(1 for o in offs if abs(o) < 2.5)
    print(f"   within +/-2.5 deg of the beam: {near_zero}/{n} = {100*near_zero/n:.1f}%")
    print(f"   -> copying the beam position would score {100*near_zero/n:.1f}% 'close'")

    # 2. does the label stay inside the learnable region
    print()
    print("2. IS EVERY LABEL LEARNABLE (peak inside the fan, not saturated)?")
    sat = 0
    for s in samples:
        peak = s["rsrp"].index(max(s["rsrp"]))
        if peak in (0, N_BEAMS - 1):
            sat += 1
    print(f"   peak sitting on an EDGE beam: {sat}/{n} = {100*sat/n:.1f}%")
    print("   (edge peaks are where the centroid starts to saturate)")

    # 3. dynamic range: is there a gradient to read at all
    print()
    print("3. IS THERE A GRADIENT TO READ?")
    drs = [max(s["rsrp"]) - min(s["rsrp"]) for s in samples]
    flat = sum(1 for d in drs if d < 3)
    print(f"   mean dynamic range {sum(drs)/n:.1f} dB   min {min(drs)} dB")
    print(f"   profiles flatter than 3 dB: {flat}/{n}")

    # 4. label spread: are we producing continuous values or buckets
    print()
    print("4. ARE THE LABELS CONTINUOUS (not bucketed)?")
    uniq = len(set(s["label"] for s in samples))
    print(f"   distinct label values: {uniq} out of {n} samples")

    # 5. the copy test, offline
    print()
    print("5. OFFLINE COPY TEST: how well would 'echo the beam position' do?")
    err_copy = sum(abs(o) for o in offs) / n
    print(f"   mean error if the model just copied the beam: {err_copy:.2f} deg")
    print(f"   (in the previous training set this was under 1 deg, which is why it copied)")
    print()


def self_test():
    """Cross-check the port against values computed from the JS on the same input."""
    print("SELF TEST: port fidelity")
    # deterministic input, no shadow fading, crowd at +10 deg / 100 m, fan at 0
    ue_az = [10.0] * 60
    ue_rng = [100.0] * 60
    ue_sh = [0.0] * 60
    rsrp = rsrp_per_beam(ue_az, ue_rng, ue_sh, 0.0)
    cen = rsrp_centroid(rsrp, 0.0)
    print(f"   beam azimuths : {[round(a,1) for a in fan_azimuths(0.0)]}")
    print(f"   rsrp          : {rsrp}")
    print(f"   centroid      : {cen:.2f} deg   (crowd at +10.0)")
    assert all(RSRP_MIN <= r <= RSRP_MAX for r in rsrp), "RSRP outside TS 38.133 range"
    assert abs(cen - 10.0) < 3.0, "centroid far from the crowd on a clean case"
    print("   OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="train_v8.jsonl")
    a = ap.parse_args()

    self_test()

    rng = random.Random(a.seed)
    samples = [make_sample(rng) for _ in range(a.n)]

    with open(a.out, "w") as f:
        for s in samples:
            f.write(json.dumps({"messages": to_messages(s)}) + "\n")

    diagnostics(samples)

    print("EXAMPLE PROMPT (this is exactly what the model sees):")
    m = to_messages(samples[0])
    print("   user  :", m[1]["content"])
    print("   answer:", m[2]["content"])
    print()
    print(f"wrote {len(samples)} samples to {a.out}")


if __name__ == "__main__":
    main()
