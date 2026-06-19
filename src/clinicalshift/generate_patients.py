"""
Generate synthetic patient cohort for ClinicalShift-2026.

Produces 50K patients with T2DM ± CKD ± HTN, realistic medication
profiles, and controlled prescribing-error rates for experimental
ground-truth contradiction generation.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Prescribing-error rate: proportion of patients with eGFR 25-30
# who remain on metformin (simulates real-world prescribing inertia)
PRESCRIBING_ERROR_RATE = 0.05


def sample_dx():
    """Sample diagnoses. CKD stages are mutually exclusive."""
    dx = ["T2DM"]
    # CKD staging: check stage 4 first (more severe), mutually exclusive
    if rng.random() < 0.20:
        dx.append("CKD_stage4")
    elif rng.random() < 0.40:
        dx.append("CKD_stage3")
    if rng.random() < 0.60:
        dx.append("HTN")
    return dx


def sample_egfr(dx_list):
    """Sample eGFR based on CKD stage."""
    base = rng.normal(70, 15)
    if "CKD_stage3" in dx_list:
        base -= 25
    if "CKD_stage4" in dx_list:
        base -= 40
    return float(max(5, min(120, base)))


def _sample_diabetes_meds(egfr, age):
    """Select glucose-lowering therapy based on renal function and age.

    Elderly (>80) get reduced-dose metformin (250mg OD) to reflect
    geriatric prescribing guidelines.
    """
    if egfr >= 30:
        if age > 80:
            # Elderly: reduced dose per geriatric guidelines
            return [{"drug": "metformin", "dose": "250mg", "freq": "od"}]
        return [{"drug": "metformin", "dose": "500mg", "freq": "bd"}]
    if egfr >= 25 and rng.random() < PRESCRIBING_ERROR_RATE:
        # Prescribing error: metformin continued despite borderline eGFR
        return [{"drug": "metformin", "dose": "500mg", "freq": "bd"}]
    return [{"drug": "insulin_glargine", "dose": "10units", "freq": "od"}]


def _sample_addon_meds(has_ckd, egfr):
    """Sample SGLT2i and DPP-4i add-on therapy."""
    result = []
    if has_ckd and egfr >= 20 and rng.random() < 0.30:
        result.append({"drug": "empagliflozin", "dose": "10mg", "freq": "od"})
    if 30 <= egfr <= 60 and rng.random() < 0.20:
        dose = "100mg" if egfr > 50 else "50mg"
        result.append({"drug": "sitagliptin", "dose": dose, "freq": "od"})
    return result


def _sample_htn_meds(has_ckd):
    """Sample antihypertensive therapy."""
    result = [{"drug": "amlodipine", "dose": "5mg", "freq": "od"}]
    if has_ckd and rng.random() < 0.40:
        result.append({"drug": "ramipril", "dose": "5mg", "freq": "od"})
    return result


def sample_medications(dx_list, egfr, age):
    """Sample medications based on diagnoses, eGFR, and age.

    Age influences prescribing: elderly patients (>80) have a higher
    chance of being on reduced-dose metformin or switched to DPP-4
    inhibitors per geriatric guidelines.
    """
    medications = []
    has_ckd = "CKD_stage3" in dx_list or "CKD_stage4" in dx_list

    if "T2DM" in dx_list:
        medications.extend(_sample_diabetes_meds(egfr, age))
        medications.extend(_sample_addon_meds(has_ckd, egfr))

    if "HTN" in dx_list:
        medications.extend(_sample_htn_meds(has_ckd))

    return medications


def generate_summary(pid, age, sex, dx_list, egfr, medications):
    """Generate a natural-language clinical summary."""
    dx_str = ", ".join(dx_list)
    meds_str = (
        ", ".join([f"{m['drug']} {m['dose']} {m['freq']}" for m in medications])
        or "no current medications"
    )
    return (
        f"Patient {pid} is a {age}-year-old {sex} with {dx_str}. "
        f"Estimated GFR is {egfr:.1f} mL/min. "
        f"Current medications include {meds_str}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient cohort."
    )
    parser.add_argument(
        "--n", type=int, default=50000,
        help="Number of patients to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    rows = []
    for pid in range(1, args.n + 1):
        age = int(np.clip(rng.normal(60, 12), 30, 95))
        sex = random.choice(["male", "female"])
        dx_list = sample_dx()
        egfr = sample_egfr(dx_list)
        medications = sample_medications(dx_list, egfr, age)
        time_epoch = random.choice([2025, 2026])
        summary_gt = generate_summary(pid, age, sex, dx_list, egfr, medications)
        rows.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "egfr": egfr,
            "dx_list": ";".join(dx_list),
            "med_list": json.dumps(medications),
            "time_epoch": time_epoch,
            "summary_gt": summary_gt,
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "patients.csv", index=False)

    # Distribution report
    on_met = df[df["med_list"].str.contains("metformin", na=False)]
    print(f"Generated {len(df)} patients -> data/patients.csv")
    print(f"  On metformin: {len(on_met)} ({100*len(on_met)/len(df):.1f}%)")
    print(f"  Metformin + eGFR<30: {len(on_met[on_met.egfr < 30])}"
          f" (prescribing errors)")
    print(f"  Metformin + eGFR 30-45: {len(on_met[(on_met.egfr>=30)&(on_met.egfr<45)])}"
          f" (tau_new contradictions)")
    print(f"  CKD_stage4 + metformin: "
          f"{len(on_met[on_met.dx_list.str.contains('CKD_stage4', na=False)])}")
    ckd_both = df[df.dx_list.str.contains("CKD_stage3") & df.dx_list.str.contains("CKD_stage4")]
    print(f"  CKD3+CKD4 co-occurrence: {len(ckd_both)} (should be 0)")


if __name__ == "__main__":
    main()
