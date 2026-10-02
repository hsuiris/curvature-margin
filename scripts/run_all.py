"""Run every analysis in dependency order (a03 before a04, e1 before e3). Usage: python scripts/run_all.py [-j 4]"""
import argparse, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAINS = [["a02_decoding_alignment", "a03_scorer_family", "a04_temperature_sweep"],
          ["e1_scaled_sweep", "e2_curvature_margin", "e3_falcon_dual"],
          ["a01_mage_families"], ["a05_blindspot_features"], ["a06_gate_stack"], ["a07_raid analyze"], ["a08_raid_compare"],
          ["a09_signal_ablation"], ["a10_temptest analyze"], ["a11_new_generators"], ["a12_paraphrase"],
          ["e4_m4gt_natural"], ["e5_heldout_pythia"], ["e6_zero_shot_benchmark"], ["e7_trained_baselines"],
          ["e8_margin_generality"], ["e9_theory_prediction"], ["e10_blind_spot_combination"], ["e11_retokenization_diagnostic"]]


def run_chain(chain):
    for item in chain:
        name, *args = item.split(); t0 = time.time()
        rc = subprocess.call([sys.executable, str(HERE / f"{name}.py"), *args], stdout=subprocess.DEVNULL)
        print(f"{name}: {'ok' if rc == 0 else f'FAILED (exit {rc})'} in {time.time() - t0:.0f}s", flush=True)
        if rc:
            return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-j", type=int, default=1, help="parallel chains"); a = ap.parse_args()
    with ThreadPoolExecutor(max_workers=a.j) as ex:
        ok = all(ex.map(run_chain, CHAINS))
    sys.exit(0 if ok else 1)
