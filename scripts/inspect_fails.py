"""Inspect failing ARC tasks. Pure viewer — no solving, no theater."""
import arckit
import sys

# First 10 same-size failures from baselines/baseline_phase_traced.txt
FAILS = [
    "045e512c", "05a7bcf2", "0607ce86", "06df4c85", "09629e4f",
    "09c534e7", "0a2355a6", "0e206a2e", "0e671a1a", "1190bc91",
]

train_set, _ = arckit.load_data()
by_id = {t.id: t for t in train_set}

def show(grid):
    for row in grid:
        print("  " + " ".join(f"{int(v)}" for v in row))

target = sys.argv[1] if len(sys.argv) > 1 else None
ids = [target] if target else FAILS

for tid in ids:
    if tid not in by_id:
        print(f"[{tid}] NOT FOUND")
        continue
    task = by_id[tid]
    print(f"\n=== {tid} ===")
    print(f"train pairs: {len(task.train)}")
    for i, (inp, out) in enumerate(task.train):
        print(f"\n-- pair {i}: in {inp.shape} -> out {out.shape}")
        print("INPUT:")
        show(inp)
        print("OUTPUT:")
        show(out)
