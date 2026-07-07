import pandas as pd
import os
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

phantom_path = os.path.join(project_root, "results", "tables", "13_per_galaxy_diagnostics.csv")
ml_path      = os.path.join(project_root, "results", "tables", "optimized_ml_ratios_eg.csv")
sparc_path   = "/Users/skrout/Documents/LaTeX/Emergent Gravity/emergent-gravity-sparc/data/Table1_Lelli2016c.mrt"


def normalize_galaxy_name(name):
    name = name.strip().replace(' ', '')
    name = re.sub(r'([A-Za-z]+)0+(\d)', r'\1\2', name)
    return name.upper()


phantom = pd.read_csv(phantom_path)
ml      = pd.read_csv(ml_path)
ml["Hit_Bounds"] = ml["Hit_Bounds"].astype(str).str.strip().str.lower() == "true"

failures      = phantom[phantom["Genuine_Failure"] == True]["Galaxy"]
boundary_hits = ml[ml["Hit_Bounds"] == True]["Galaxy"]

overlap = set(failures) & set(boundary_hits)
print("\n--- M/L BOUNDARY OVERLAP ---")
print(f"Overlap: {len(overlap)} of {len(failures)} genuine failures also hit M/L bounds")
print(f"Galaxies: {overlap}")

print("\n--- SPARC Q-FLAG DISTRIBUTION ---")

failures_normalized = {normalize_galaxy_name(g): g for g in failures.values}
q_flags = {}

print("\n[DEBUG] Sample MRT name normalization:")
debug_count = 0
with open(sparc_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        if not re.search(r'[A-Za-z]', parts[0]):
            continue
        norm = normalize_galaxy_name(parts[0])
        print(f"  raw='{parts[0]}' -> normalized='{norm}' | "
              f"in failures={norm in failures_normalized} | "
              f"token[4]='{parts[4] if len(parts) > 4 else '?'}'")
        debug_count += 1
        if debug_count >= 10:
            break

with open(sparc_path, 'r') as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) < 10:
            continue
        if not re.search(r'[A-Za-z]', parts[0]):
            continue

        mrt_name_norm = normalize_galaxy_name(parts[0])

        if mrt_name_norm in failures_normalized:
            original_name = failures_normalized[mrt_name_norm]
            try:
                q_val = int(parts[-2])
                q_flags[original_name] = q_val
            except (ValueError, IndexError):
                pass

print(f"\nQ-flags read for {len(q_flags)} of {len(failures)} failing galaxies.")
if len(q_flags) < len(failures):
    missing = set(failures.values) - set(q_flags.keys())
    print(f"Still missing: {missing}")

q_counts = {1: 0, 2: 0, 3: 0}
for g, q in q_flags.items():
    if q in q_counts:
        q_counts[q] += 1

total_read = len(q_flags)
if total_read > 0:
    print("\nQ-flag distribution for the 32 genuine failures:")
    for q_val, count in q_counts.items():
        print(f"  Q={q_val}: {count} galaxies ({(count / total_read) * 100:.1f}%)")
else:
    print("No Q-flags matched -- check debug output above for normalization issues.")