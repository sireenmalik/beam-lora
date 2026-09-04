"""
inspect_lora.py
---------------
Look inside your trained LoRA adapter file (./beam-lora/adapter_model.safetensors).

Prints:
  1. Total tensor count and total trainable params
  2. First 10 tensors with shapes and weight statistics
  3. A 3x3 sample block from one A matrix and one B matrix
  4. Layers ranked by std deviation (which layers moved most during training)
  5. Also dumps everything to lora_weights_summary.txt for scanning in Excel

Run from C:\\beam-lora with the venv active:
  .venv\\Scripts\\activate
  python inspect_lora.py
"""

import os
from safetensors.torch import load_file
import torch

ADAPTER_PATH = "./beam-lora/adapter_model.safetensors"
SUMMARY_FILE = "lora_weights_summary.txt"

# ------------------------------------------------------------------ LOAD
if not os.path.exists(ADAPTER_PATH):
    print(f"ERROR: {ADAPTER_PATH} not found.")
    print("Make sure you are in C:\\beam-lora and training finished cleanly.")
    raise SystemExit(1)

weights = load_file(ADAPTER_PATH)

# ------------------------------------------------------------------ HEADLINE STATS
total_tensors = len(weights)
total_params = sum(t.numel() for t in weights.values())
file_size_mb = os.path.getsize(ADAPTER_PATH) / (1024 * 1024)
first_dtype = list(weights.values())[0].dtype

print("=" * 70)
print("LoRA ADAPTER SUMMARY")
print("=" * 70)
print(f"File           : {ADAPTER_PATH}")
print(f"File size      : {file_size_mb:.2f} MB")
print(f"Total tensors  : {total_tensors}")
print(f"Total params   : {total_params:,}")
print(f"Dtype          : {first_dtype}")
print()

# ------------------------------------------------------------------ FIRST 10
print("=" * 70)
print("FIRST 10 TENSORS")
print("=" * 70)
for i, (name, tensor) in enumerate(weights.items()):
    if i >= 10:
        break
    print(f"  {name}")
    print(f"    shape={tuple(tensor.shape)}  dtype={tensor.dtype}  "
          f"mean={tensor.mean().item():+.5f}  std={tensor.std().item():.5f}  "
          f"min={tensor.min().item():+.4f}  max={tensor.max().item():+.4f}")
print()

# ------------------------------------------------------------------ SAMPLE BLOCKS
print("=" * 70)
print("SAMPLE 3x3 BLOCKS (actual numbers, not summaries)")
print("=" * 70)

first_A_name = next((n for n in weights if "lora_A" in n), None)
first_B_name = next((n for n in weights if "lora_B" in n), None)

if first_A_name:
    print(f"\n{first_A_name}  (down-projection, rank x in_dim)")
    print(f"shape={tuple(weights[first_A_name].shape)}")
    print("first 3x3 block:")
    print(weights[first_A_name][:3, :3])

if first_B_name:
    print(f"\n{first_B_name}  (up-projection, out_dim x rank)")
    print(f"shape={tuple(weights[first_B_name].shape)}")
    print("first 3x3 block:")
    print(weights[first_B_name][:3, :3])
print()

# ------------------------------------------------------------------ LAYERS RANKED BY STD
print("=" * 70)
print("TOP 10 LAYERS BY WEIGHT STD (which layers learned most)")
print("=" * 70)
ranked = sorted(weights.items(), key=lambda kv: kv[1].std().item(), reverse=True)
for name, tensor in ranked[:10]:
    print(f"  std={tensor.std().item():.5f}  {name}")

print()
print("BOTTOM 5 (learned least)")
for name, tensor in ranked[-5:]:
    print(f"  std={tensor.std().item():.5f}  {name}")
print()

# ------------------------------------------------------------------ DUMP TO TXT
with open(SUMMARY_FILE, "w") as f:
    f.write("name\tshape\tdtype\tparams\tmean\tstd\tmin\tmax\n")
    for name, tensor in weights.items():
        f.write(f"{name}\t{tuple(tensor.shape)}\t{tensor.dtype}\t"
                f"{tensor.numel()}\t"
                f"{tensor.mean().item():+.6f}\t{tensor.std().item():.6f}\t"
                f"{tensor.min().item():+.6f}\t{tensor.max().item():+.6f}\n")

print("=" * 70)
print(f"Full listing written to: {SUMMARY_FILE}")
print("Open in Excel to sort/filter all 336 tensors.")
print("=" * 70)
