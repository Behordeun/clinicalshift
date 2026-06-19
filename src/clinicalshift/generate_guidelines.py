"""
Generate clinical guideline collections for ClinicalShift-2026.

Produces 60 guidelines per collection across 5 clinical categories,
with controlled temporal shift (~30%), institutional vocabulary swap,
and schema erasure variants.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# =====================================================================
# Vocabulary map for institutional shift (instA → instB)
# =====================================================================

VOCAB_MAP = {
    "metformin": "biguanide class agent",
    "Metformin": "Biguanide class agent",
    "eGFR": "estimated renal clearance rate",
    "mL/min": "mL/min/1.73m\u00b2",
    "CKD": "chronic renal insufficiency (CRI)",
    "CKD stage 3": "CRI grade III",
    "CKD stage 4": "CRI grade IV",
    "CKD stage 5": "CRI grade V",
    "stage 3a": "grade IIIa",
    "stage 3b": "grade IIIb",
    "stage 4": "grade IV",
    "stage 5": "grade V",
    "T2DM": "non-insulin-dependent diabetes mellitus (NIDDM)",
    "type 2 diabetes": "non-insulin-dependent diabetes",
    "HTN": "arterial hypertension",
    "HbA1c": "glycated haemoglobin",
    "amlodipine": "dihydropyridine calcium antagonist",
    "ramipril": "angiotensin-converting enzyme inhibitor",
    "ACE inhibitor": "angiotensin-converting enzyme inhibitor",
    "ACE inhibitors": "angiotensin-converting enzyme inhibitors",
    "ARB": "angiotensin receptor blocker",
    "ARBs": "angiotensin receptor blockers",
    "SGLT2 inhibitor": "sodium-glucose co-transporter 2 blocker",
    "SGLT2 inhibitors": "sodium-glucose co-transporter 2 blockers",
    "SGLT2": "SGLT2",
}


# =====================================================================
# Baseline guidelines (tau_old, 2025, Institution A) — 60 documents
# =====================================================================

BASELINE_GUIDELINES: List[Dict[str, Any]] = [
    # --- T2DM Pharmacotherapy (20 docs) ---
    {
        "id": "t2dm_01",
        "text": "For patients with type 2 diabetes and eGFR greater than or equal to 30 mL/min, metformin is recommended as first-line glucose-lowering therapy at a starting dose of 500mg twice daily.",
        "tags": ["T2DM", "metformin", "firstline"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_02",
        "text": "Metformin is contraindicated in patients with type 2 diabetes when eGFR falls below 30 mL/min due to increased risk of lactic acidosis. Insulin therapy should be initiated as an alternative.",
        "tags": ["T2DM", "CKD", "metformin", "contraindication"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_03",
        "text": "Metformin dosing: initiate at 500mg twice daily with meals. Titrate by 500mg weekly to a maximum of 2000mg daily as tolerated. Extended-release formulations may improve gastrointestinal tolerance.",
        "tags": ["T2DM", "metformin", "dosing"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_04",
        "text": "Metformin is absolutely contraindicated in CKD stage 4 or stage 5 regardless of current eGFR value, due to severely impaired renal clearance and unacceptable risk of lactic acidosis accumulation.",
        "tags": ["T2DM", "CKD", "metformin", "contraindication"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_05",
        "text": "Common gastrointestinal side effects of metformin include nausea, diarrhoea, and abdominal discomfort. These can be minimised by taking the medication with food and using slow-release preparations.",
        "tags": ["T2DM", "metformin", "adverse_effects"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_06",
        "text": "Insulin glargine should be initiated at 10 units once daily at bedtime. Titrate upward by 2 units every 3 days until fasting glucose target of 4.0-7.0 mmol/L is achieved.",
        "tags": ["T2DM", "insulin", "dosing"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_07",
        "text": "Insulin glargine is the preferred agent when eGFR is below 30 mL/min or when HbA1c remains above 9% despite maximum tolerated doses of two oral glucose-lowering agents.",
        "tags": ["T2DM", "insulin", "initiation"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_08",
        "text": "SGLT2 inhibitors such as empagliflozin and dapagliflozin are recommended as add-on therapy in patients with type 2 diabetes who have established cardiovascular disease or are at high cardiovascular risk.",
        "tags": ["T2DM", "SGLT2", "cardiovascular"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_09",
        "text": "SGLT2 inhibitors should not be initiated when eGFR is below 20 mL/min. For patients already established on therapy, discontinuation is recommended when eGFR falls below 15 mL/min.",
        "tags": ["T2DM", "SGLT2", "contraindication"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_10",
        "text": "DPP-4 inhibitors such as sitagliptin require renal dose adjustment: 100mg daily if eGFR above 50, 50mg daily if eGFR 30-50, and 25mg daily if eGFR below 30 mL/min.",
        "tags": ["T2DM", "DPP4", "dosing", "CKD"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_11",
        "text": "Sulfonylureas such as gliclazide should be avoided in patients older than 75 years due to the increased risk of severe and prolonged hypoglycaemia in the elderly population.",
        "tags": ["T2DM", "sulfonylurea", "elderly"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_12",
        "text": "GLP-1 receptor agonists including liraglutide are recommended as second-line agents after metformin. They do not require renal dose adjustment and provide cardiovascular and weight benefits.",
        "tags": ["T2DM", "GLP1"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_13",
        "text": "The target HbA1c for most adults with type 2 diabetes is below 7.0% (53 mmol/mol). Individualised targets should account for hypoglycaemia risk, duration of diabetes, and comorbidities.",
        "tags": ["T2DM", "target", "HbA1c"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_14",
        "text": "For elderly patients over 75 years or those with significant comorbidities, a relaxed HbA1c target of below 8.0% (64 mmol/mol) is appropriate to minimise hypoglycaemia risk.",
        "tags": ["T2DM", "target", "HbA1c", "elderly"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_15",
        "text": "When combination therapy is needed, metformin plus an SGLT2 inhibitor is preferred over metformin plus a sulfonylurea due to lower hypoglycaemia risk and cardiovascular benefit.",
        "tags": ["T2DM", "combination"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_16",
        "text": "Insulin therapy should be initiated when HbA1c remains above 9% despite 3 months of optimised dual oral therapy. Basal insulin added to oral agents is the preferred starting regimen.",
        "tags": ["T2DM", "insulin", "initiation"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_17",
        "text": "Hypoglycaemia management: administer 15 grams of fast-acting glucose orally. Recheck blood glucose in 15 minutes and repeat treatment if below 4.0 mmol/L.",
        "tags": ["T2DM", "hypoglycemia"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_18",
        "text": "Sick day rules for metformin: temporarily withhold metformin during periods of acute illness, dehydration, vomiting, or surgical fasting. Resume 48 hours after recovery.",
        "tags": ["T2DM", "metformin", "sick_day"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_19",
        "text": "All patients with type 2 diabetes should undergo annual screening for diabetic retinopathy through dilated fundoscopy or validated digital retinal photography.",
        "tags": ["T2DM", "screening"],
        "category": "t2dm",
    },
    {
        "id": "t2dm_20",
        "text": "Annual comprehensive foot examination is required for all patients with type 2 diabetes, including monofilament sensory testing, assessment of pedal pulses, and inspection for ulceration.",
        "tags": ["T2DM", "screening"],
        "category": "t2dm",
    },
    # --- CKD Management (15 docs) ---
    {
        "id": "ckd_01",
        "text": "CKD is staged by eGFR: stage 1 (eGFR above 90), stage 2 (60-89), stage 3a (45-59), stage 3b (30-44), stage 4 (15-29), and stage 5 (below 15 mL/min).",
        "tags": ["CKD", "staging"],
        "category": "ckd",
    },
    {
        "id": "ckd_02",
        "text": "Monitor eGFR every 3 months in patients with an annual decline exceeding 5 mL/min/year. Stable CKD stage 3 requires monitoring every 6 to 12 months.",
        "tags": ["CKD", "monitoring"],
        "category": "ckd",
    },
    {
        "id": "ckd_03",
        "text": "Renal dose adjustment is required for the majority of renally cleared medications when eGFR falls below 45 mL/min. Consult pharmacy for individual drug recommendations.",
        "tags": ["CKD", "dosing"],
        "category": "ckd",
    },
    {
        "id": "ckd_04",
        "text": "NSAIDs and aminoglycosides should be avoided in patients with CKD due to the risk of precipitating acute kidney injury and accelerating disease progression.",
        "tags": ["CKD", "nephrotoxic"],
        "category": "ckd",
    },
    {
        "id": "ckd_05",
        "text": "ACE inhibitors such as ramipril are renoprotective in CKD with proteinuria. Initiate at low dose, uptitrate to maximum tolerated, and monitor serum potassium and creatinine.",
        "tags": ["CKD", "ACEi", "renoprotection"],
        "category": "ckd",
    },
    {
        "id": "ckd_06",
        "text": "Serum potassium should be checked within 1 week of initiating or uptitrating ACE inhibitors or ARBs in patients with CKD. Withhold if potassium exceeds 6.0 mmol/L.",
        "tags": ["CKD", "monitoring", "potassium"],
        "category": "ckd",
    },
    {
        "id": "ckd_07",
        "text": "Investigate anaemia in CKD when haemoglobin falls below 100 g/L. Check serum ferritin and transferrin saturation. Consider erythropoiesis-stimulating agents if iron-replete.",
        "tags": ["CKD", "anaemia"],
        "category": "ckd",
    },
    {
        "id": "ckd_08",
        "text": "CKD-mineral bone disease management: initiate phosphate binders and vitamin D supplementation from stage 3b onwards. Monitor calcium, phosphate, and PTH quarterly.",
        "tags": ["CKD", "bone"],
        "category": "ckd",
    },
    {
        "id": "ckd_09",
        "text": "Refer to nephrology when eGFR falls below 30 mL/min, or if there is a sustained decline exceeding 5 mL/min per year, or if there is significant proteinuria (ACR above 70 mg/mmol).",
        "tags": ["CKD", "referral"],
        "category": "ckd",
    },
    {
        "id": "ckd_10",
        "text": "To prevent contrast-induced nephropathy: pre-hydrate with 0.9% sodium chloride, use low-osmolar contrast, and withhold metformin for 48 hours following contrast administration.",
        "tags": ["CKD", "contrast", "metformin"],
        "category": "ckd",
    },
    {
        "id": "ckd_11",
        "text": "Arteriovenous fistula creation should be planned when eGFR approaches 20 mL/min in patients likely to require haemodialysis, allowing 6 months for maturation.",
        "tags": ["CKD", "dialysis"],
        "category": "ckd",
    },
    {
        "id": "ckd_12",
        "text": "CKD confers cardiovascular risk equivalent to established coronary artery disease. All CKD patients should receive secondary-prevention-intensity statin therapy and antiplatelet assessment.",
        "tags": ["CKD", "cardiovascular"],
        "category": "ckd",
    },
    {
        "id": "ckd_13",
        "text": "Fluid restriction to 1.5 litres per day is advised in advanced CKD (stage 4-5) with clinical evidence of fluid overload or peripheral oedema.",
        "tags": ["CKD", "fluid"],
        "category": "ckd",
    },
    {
        "id": "ckd_14",
        "text": "Drug clearance is significantly affected in CKD. Metformin, gabapentin, and digoxin all require eGFR-based dose adjustment. Review all medications at each CKD stage transition.",
        "tags": ["CKD", "dosing", "clearance"],
        "category": "ckd",
    },
    {
        "id": "ckd_15",
        "text": "Metabolic acidosis in CKD should be treated with oral sodium bicarbonate supplementation when serum bicarbonate falls below 22 mmol/L, targeting a level of 22-26 mmol/L.",
        "tags": ["CKD", "acidosis"],
        "category": "ckd",
    },
    # --- HTN Co-Management (12 docs) ---
    {
        "id": "htn_01",
        "text": "Blood pressure target for patients with type 2 diabetes is below 130/80 mmHg. More aggressive targets (below 120/80) may be considered in high-risk patients without orthostatic symptoms.",
        "tags": ["HTN", "target", "T2DM"],
        "category": "htn",
    },
    {
        "id": "htn_02",
        "text": "First-line antihypertensive therapy for patients with type 2 diabetes should be an ACE inhibitor or ARB, particularly when albuminuria or proteinuria is present.",
        "tags": ["HTN", "firstline", "T2DM"],
        "category": "htn",
    },
    {
        "id": "htn_03",
        "text": "Calcium channel blockers such as amlodipine 5-10mg daily are appropriate second-line therapy, or first-line in patients intolerant of ACE inhibitors due to cough or angioedema.",
        "tags": ["HTN", "CCB"],
        "category": "htn",
    },
    {
        "id": "htn_04",
        "text": "In patients with CKD, preferred antihypertensive combination is ACE inhibitor or ARB plus calcium channel blocker. Avoid ACEi/ARB plus thiazide in advanced CKD due to electrolyte disturbance.",
        "tags": ["HTN", "combination", "CKD"],
        "category": "htn",
    },
    {
        "id": "htn_05",
        "text": "For resistant hypertension (uncontrolled on three agents including a diuretic), add spironolactone 25mg daily. Monitor potassium closely, especially in CKD.",
        "tags": ["HTN", "resistant"],
        "category": "htn",
    },
    {
        "id": "htn_06",
        "text": "In CKD with proteinuria: ACE inhibitor or ARB is mandatory for renoprotection regardless of blood pressure level. Target BP below 130/80 mmHg.",
        "tags": ["HTN", "CKD", "proteinuria"],
        "category": "htn",
    },
    {
        "id": "htn_07",
        "text": "Do not combine ACE inhibitor with ARB (dual RAAS blockade). This combination increases risk of hyperkalaemia, acute kidney injury, and hypotension without additional renal benefit.",
        "tags": ["HTN", "contraindication"],
        "category": "htn",
    },
    {
        "id": "htn_08",
        "text": "Beta-blockers should be used cautiously in patients with diabetes as they may mask adrenergic symptoms of hypoglycaemia and impair glycaemic awareness.",
        "tags": ["HTN", "beta_blocker", "T2DM"],
        "category": "htn",
    },
    {
        "id": "htn_09",
        "text": "Home blood pressure monitoring is preferred over clinic readings for diagnosis and treatment titration. Advise twice-daily measurements for 7 days before clinical review.",
        "tags": ["HTN", "monitoring"],
        "category": "htn",
    },
    {
        "id": "htn_10",
        "text": "Assess for orthostatic hypotension by measuring standing blood pressure in all elderly patients on antihypertensives. A drop exceeding 20/10 mmHg warrants treatment review.",
        "tags": ["HTN", "elderly"],
        "category": "htn",
    },
    {
        "id": "htn_11",
        "text": "Screen for renal artery stenosis in patients with resistant hypertension plus unexplained deterioration in renal function, particularly after ACE inhibitor initiation.",
        "tags": ["HTN", "secondary"],
        "category": "htn",
    },
    {
        "id": "htn_12",
        "text": "Hypertensive emergency (BP above 180/120 with end-organ damage): initiate IV labetalol and target 25% blood pressure reduction within the first hour.",
        "tags": ["HTN", "emergency"],
        "category": "htn",
    },
    # --- Age/Dosing Specific (8 docs) ---
    {
        "id": "age_01",
        "text": "For elderly patients over 75 years: target HbA1c below 8.0%, avoid sulfonylureas, prefer DPP-4 inhibitors or basal insulin for glucose lowering with lower hypoglycaemia risk.",
        "tags": ["elderly", "T2DM", "target"],
        "category": "dosing",
    },
    {
        "id": "age_02",
        "text": "In patients over 80 years, avoid aggressive blood pressure lowering. Target below 150/90 mmHg and assess for falls risk before initiating or intensifying antihypertensives.",
        "tags": ["elderly", "HTN"],
        "category": "dosing",
    },
    {
        "id": "age_03",
        "text": "Conduct structured medication review (polypharmacy assessment) in all patients taking more than 5 regular medications. Deprescribe where risk outweighs benefit.",
        "tags": ["elderly", "polypharmacy"],
        "category": "dosing",
    },
    {
        "id": "age_04",
        "text": "Falls risk is significantly increased with antihypertensives, sedatives, and hypoglycaemic agents in elderly patients. Perform annual falls risk assessment using validated tools.",
        "tags": ["elderly", "falls"],
        "category": "dosing",
    },
    {
        "id": "age_05",
        "text": "Metformin in elderly patients: initiate at 250mg once daily, titrate slowly to maximum 1500mg daily. Monitor vitamin B12 annually as metformin impairs absorption.",
        "tags": ["elderly", "metformin", "dosing"],
        "category": "dosing",
    },
    {
        "id": "age_06",
        "text": "Age-related decline in eGFR of approximately 1 mL/min per year after age 40 is physiological and does not necessarily indicate CKD progression if stable and without proteinuria.",
        "tags": ["elderly", "CKD", "normal_decline"],
        "category": "dosing",
    },
    {
        "id": "age_07",
        "text": "Use the Clinical Frailty Scale to assess frailty before intensifying any treatment in patients over 75. Frail patients (CFS 6-7) benefit from simplified regimens.",
        "tags": ["elderly", "frailty"],
        "category": "dosing",
    },
    {
        "id": "age_08",
        "text": "End-of-life diabetes management: simplify to basal insulin only, discontinue metformin and monitoring, prioritise symptom control and avoidance of hypoglycaemia.",
        "tags": ["elderly", "palliative"],
        "category": "dosing",
    },
    # --- Distractors (5 docs) ---
    {
        "id": "dist_01",
        "text": "COPD management: initiate LABA/LAMA combination inhaler as first-line maintenance therapy. Avoid prolonged systemic corticosteroids due to hyperglycaemia and osteoporosis risk.",
        "tags": ["COPD"],
        "category": "distractor",
    },
    {
        "id": "dist_02",
        "text": "Annual thyroid function screening with TSH is recommended for all patients with type 2 diabetes due to increased prevalence of autoimmune thyroid disease.",
        "tags": ["thyroid", "screening"],
        "category": "distractor",
    },
    {
        "id": "dist_03",
        "text": "Screen for depression using PHQ-9 annually in all patients with chronic disease including diabetes and CKD. Treat with SSRI if score 10 or above.",
        "tags": ["depression", "screening"],
        "category": "distractor",
    },
    {
        "id": "dist_04",
        "text": "Statin therapy with atorvastatin 20mg daily is recommended for primary cardiovascular prevention in all patients with type 2 diabetes aged over 40 years.",
        "tags": ["statin", "cardiovascular"],
        "category": "distractor",
    },
    {
        "id": "dist_05",
        "text": "Pneumococcal polysaccharide vaccination (PPV23) is recommended for all patients with CKD stage 3 or above, and should be repeated every 5 years.",
        "tags": ["vaccination", "CKD"],
        "category": "distractor",
    },
]

# =====================================================================
# Temporal shift overrides (tau_new, 2026) — 18 docs modified
# =====================================================================

TEMPORAL_SHIFTS: Dict[str, str] = {
    "t2dm_01": "For patients with type 2 diabetes and eGFR greater than or equal to 45 mL/min, metformin is recommended as first-line glucose-lowering therapy. For eGFR 30-44, consider SGLT2 inhibitors as first-line instead.",
    "t2dm_02": "Metformin is contraindicated in patients with type 2 diabetes when eGFR falls below 45 mL/min due to increased risk of lactic acidosis. SGLT2 inhibitors or insulin therapy should be initiated as alternatives.",
    "t2dm_08": "SGLT2 inhibitors such as empagliflozin and dapagliflozin are now recommended as first-line therapy for patients with type 2 diabetes and CKD, providing both glycaemic control and renoprotective benefit.",
    "t2dm_09": "SGLT2 inhibitors should not be initiated when eGFR is below 15 mL/min. For patients already established on therapy, continue until dialysis initiation.",
    "t2dm_10": "DPP-4 inhibitors such as sitagliptin require renal dose adjustment: 100mg daily if eGFR above 50, 50mg daily if eGFR 30-60, and 25mg daily if eGFR below 30 mL/min.",
    "t2dm_13": "The target HbA1c for most adults with type 2 diabetes is below 7.0% (53 mmol/mol). For patients with CKD, an SGLT2-inhibitor-first strategy is now preferred regardless of HbA1c level.",
    "t2dm_15": "For patients with CKD, SGLT2 inhibitor monotherapy is now preferred over metformin-based combinations due to superior renoprotective evidence and lower lactic acidosis risk.",
    "t2dm_18": "Sick day rules for metformin: temporarily withhold metformin during periods of acute illness, dehydration, vomiting, or surgical fasting. Resume 72 hours after full recovery and confirm eGFR stability.",
    "ckd_03": "Renal dose adjustment is required for the majority of renally cleared medications when eGFR falls below 60 mL/min. More aggressive adjustment is needed below 30 mL/min.",
    "ckd_05": "SGLT2 inhibitors are now the preferred first-line renoprotective agent in CKD with proteinuria, with ACE inhibitors as second-line. Combination SGLT2 plus ACEi provides additive benefit.",
    "ckd_09": "Refer to nephrology when eGFR falls below 45 mL/min, or if there is a sustained decline exceeding 5 mL/min per year, or if there is significant proteinuria (ACR above 30 mg/mmol).",
    "ckd_10": "To prevent contrast-induced nephropathy: pre-hydrate with 0.9% sodium chloride, use low-osmolar contrast, and withhold metformin for 72 hours following contrast administration.",
    "ckd_14": "Drug clearance is significantly affected in CKD. Metformin, SGLT2 inhibitors, gabapentin, and digoxin all require eGFR-based dose adjustment. Review all medications at each CKD stage transition.",
    "htn_04": "In patients with CKD, preferred antihypertensive combination is SGLT2 inhibitor plus ACE inhibitor or ARB plus calcium channel blocker as triple therapy for comprehensive reno-cardiovascular protection.",
    "htn_06": "In CKD with proteinuria: SGLT2 inhibitor is now mandatory alongside ACE inhibitor or ARB for renoprotection. Target BP below 130/80 mmHg.",
    "age_01": "For elderly patients over 75 years: target HbA1c below 7.5%, avoid sulfonylureas, prefer SGLT2 inhibitors if eGFR above 45, otherwise DPP-4 inhibitors or basal insulin.",
    "age_05": "Metformin in elderly patients: avoid if eGFR below 45 mL/min. When used, initiate at 250mg once daily, maximum 1000mg daily. Monitor vitamin B12 and lactate annually.",
    "dist_04": "Statin therapy with atorvastatin 40mg daily is recommended for primary cardiovascular prevention in all patients with type 2 diabetes aged over 40 years. High-intensity statin is now standard.",
}

# =====================================================================
# Generation functions
# =====================================================================


def apply_vocab_transform(text: str, vocab_map: Dict[str, str]) -> str:
    """Apply vocabulary substitution, longest-match-first to avoid partial replacements."""
    # Sort by length descending to replace longer phrases first
    sorted_terms = sorted(vocab_map.keys(), key=len, reverse=True)
    for term in sorted_terms:
        text = text.replace(term, vocab_map[term])
    return text


def generate_collection(
    base_guidelines: List[Dict[str, Any]],
    temporal_shifts: Dict[str, str] | None = None,
    vocab_map: Dict[str, str] | None = None,
    epoch: str = "2025",
    institution: str = "A",
    strip_metadata: bool = False,
) -> List[Dict[str, Any]]:
    """Generate a guideline collection with optional shifts applied."""
    collection = []
    for g in base_guidelines:
        doc = dict(g)
        doc["epoch"] = epoch
        doc["institution"] = institution

        # Apply temporal shift if this doc has an override
        if temporal_shifts and doc["id"] in temporal_shifts:
            doc["text"] = temporal_shifts[doc["id"]]
            doc["version"] = "2026_updated"
        else:
            doc["version"] = "2025_original"

        # Apply vocabulary transformation
        if vocab_map:
            doc["text"] = apply_vocab_transform(doc["text"], vocab_map)

        # Strip metadata for schema erasure
        if strip_metadata:
            doc = {"id": doc["id"], "text": doc["text"]}

        collection.append(doc)
    return collection


def main():
    parser = argparse.ArgumentParser(
        description="Generate clinical guideline collections."
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DATA_DIR),
        help="Output directory for JSON files",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # tau_old: baseline 2025, institution A
    tau_old = generate_collection(
        BASELINE_GUIDELINES, epoch="2025", institution="A"
    )
    # tau_new: 2026 with temporal shifts
    tau_new = generate_collection(
        BASELINE_GUIDELINES, temporal_shifts=TEMPORAL_SHIFTS,
        epoch="2026", institution="A"
    )
    # instA: same as tau_old
    inst_a = tau_old
    # instB: vocabulary-transformed
    inst_b = generate_collection(
        BASELINE_GUIDELINES, vocab_map=VOCAB_MAP,
        epoch="2025", institution="B"
    )
    # schema_erased: tau_old content with metadata stripped
    schema_erased = generate_collection(
        BASELINE_GUIDELINES, epoch="2025", institution="A",
        strip_metadata=True,
    )

    (out / "guidelines_tau_old.json").write_text(json.dumps(tau_old, indent=2))
    (out / "guidelines_tau_new.json").write_text(json.dumps(tau_new, indent=2))
    (out / "guidelines_instA.json").write_text(json.dumps(inst_a, indent=2))
    (out / "guidelines_instB.json").write_text(json.dumps(inst_b, indent=2))
    (out / "guidelines_schema_erased.json").write_text(
        json.dumps(schema_erased, indent=2)
    )

    # Report
    n_shifted = len(TEMPORAL_SHIFTS)
    print(f"Generated guideline collections in {out}/")
    print(f"  tau_old:        {len(tau_old)} docs (baseline 2025)")
    print(f"  tau_new:        {len(tau_new)} docs ({n_shifted} shifted, "
          f"{n_shifted/len(tau_old)*100:.0f}% contradiction density)")
    print(f"  instA:          {len(inst_a)} docs (= tau_old)")
    print(f"  instB:          {len(inst_b)} docs (vocabulary transformed)")
    print(f"  schema_erased:  {len(schema_erased)} docs (metadata stripped)")


if __name__ == "__main__":
    main()
