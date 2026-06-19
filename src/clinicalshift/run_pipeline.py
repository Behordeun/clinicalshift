import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import pandas as pd
import requests
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent  # src/clinicalshift/ → src/ → project root
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PATIENTS_PATH = DATA_DIR / "patients.csv"

# Directory where your ChromaDB collections live
CHROMA_DIR = ROOT / "data" / "chroma_db"

# ---------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

VALID_REGIMES = {
    "baseline_tau_old": "tau_old",
    "temporal_drift_tau_new": "tau_new",
    "institution_swap_instB": "instB",
    "schema_erasure": "schema_erased",
}

# Regime-specific eGFR thresholds for metformin contraindication
REGIME_EGFR_THRESHOLDS = {
    "baseline_tau_old": 30,
    "temporal_drift_tau_new": 45,
    "institution_swap_instB": 30,
    "schema_erasure": 30,
}

VALID_MODES = ["single", "linear", "graph"]


# ---------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------


def prepare_patient_row(row: pd.Series, regime: str) -> pd.Series:
    """
    Transform raw patients.csv row into the format expected by run_once.

    Generates regime-dependent ground-truth contradictions:
    - metformin_egfr_below_threshold: on metformin AND eGFR < regime threshold
    - metformin_ckd_stage4: on metformin AND has CKD_stage4
    """
    import json as _json
    import re as _re

    x_raw = row["summary_gt"]
    patient_id = row["patient_id"]
    time_epoch = row["time_epoch"]

    # Parse structured fields for ground-truth contradiction generation
    egfr = float(row["egfr"])
    dx_list = str(row.get("dx_list", "")).split(";")
    med_list = _json.loads(row["med_list"]) if pd.notna(row["med_list"]) else []

    # Determine ground-truth contradictions for this regime
    gt_contradictions = set()
    uses_metformin = any(
        m.get("drug", "").lower() == "metformin" for m in med_list
    )
    has_ckd_stage4 = "CKD_stage4" in dx_list

    if uses_metformin:
        threshold = REGIME_EGFR_THRESHOLDS.get(regime, 30)
        if egfr < threshold:
            gt_contradictions.add("metformin_egfr_below_threshold")
        if has_ckd_stage4:
            gt_contradictions.add("metformin_ckd_stage4")

    # Determine epoch alignment (for TCA metric)
    # A patient is epoch-aligned if their current medications don't violate
    # the regime's guidelines
    epoch_aligned = len(gt_contradictions) == 0

    prepared = pd.Series({
        "patientid": patient_id,
        "timeepoch": time_epoch,
        "Xraw": x_raw,
        "y_star": x_raw,  # Ground truth = original factual summary
        "gtcontradictions": _json.dumps(sorted(gt_contradictions)),
        "epoch_aligned": epoch_aligned,
        "egfr": egfr,
        "dx_list": ";".join(dx_list),
        "med_list": row["med_list"],
    })
    return prepared


@dataclass
class GlobalState:
    x_raw: str
    x_parsed: Optional[Dict[str, Any]] = None
    retrieved: Optional[List[Dict[str, Any]]] = None
    audit: Optional[Dict[str, Any]] = None
    y_final: Optional[str] = None


@dataclass
class RunRecord:
    patientid: Any
    timeepoch: Any
    y_final: str
    y_star: str
    gtcontradictions: str
    auditissues: str
    confidence: float
    epoch_aligned: bool
    latency: float
    condition: str


# ---------------------------------------------------------------------
# Data + Chroma helpers
# ---------------------------------------------------------------------


def load_patients(path: Path) -> pd.DataFrame:
    """
    Expected columns: patientid, timeepoch, Xraw, y_star, gtcontradictions.
    """
    return pd.read_csv(path)


_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Singleton Chroma persistent client.
    """
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(allow_reset=False),
        )
    return _chroma_client


def get_collection_for_regime(regime: str):
    """
    Retrieve the Chroma collection associated with a shift regime.
    Collection names are assumed to match the regime keys.
    """
    client = get_chroma_client()
    return client.get_collection(name=regime)


# ---------------------------------------------------------------------
# Ollama LLM wrapper
# ---------------------------------------------------------------------


def llm_complete(prompt: str) -> str:
    """
    Call a local Ollama model to generate a completion.

    Uses OLLAMA_MODEL and OLLAMA_BASE_URL defined above.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.RequestException as exc:
        return f"[LLM ERROR: {exc}] {prompt[-400:]}"


# ---------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------


def parse_agent_call(x_raw: str) -> Dict[str, Any]:
    """
    Parse a free-text clinical note into a structured schema.

    Extracts diagnoses, medications, eGFR, and age via deterministic
    rule-based parsing. This ensures zero parser-induced hallucinations
    for Phase 1 experiments.
    """
    import re as _re

    entities: Dict[str, Any] = {"text": x_raw}

    lower = x_raw.lower()
    if "t2dm" in lower or "type 2 diabetes" in lower:
        entities.setdefault("diagnoses", []).append("T2DM")
    if "htn" in lower or "hypertension" in lower:
        entities.setdefault("diagnoses", []).append("HTN")
    if "ckd_stage4" in lower:
        entities.setdefault("diagnoses", []).append("CKD_stage4")
    if "ckd_stage3" in lower:
        entities.setdefault("diagnoses", []).append("CKD_stage3")
    if "ckd" in lower and "ckd_stage" not in lower:
        entities.setdefault("diagnoses", []).append("CKD")
    if "metformin" in lower:
        entities.setdefault("medications", []).append("metformin")
    if "insulin" in lower:
        entities.setdefault("medications", []).append("insulin_glargine")
    if "amlodipine" in lower:
        entities.setdefault("medications", []).append("amlodipine")

    # Extract eGFR (numeric) — format: "GFR is XX.X mL/min"
    egfr_match = _re.search(r"GFR is (\d+\.?\d*)", x_raw)
    if egfr_match:
        entities["egfr"] = float(egfr_match.group(1))

    # Extract age — format: "XX-year-old"
    age_match = _re.search(r"(\d+)-year-old", x_raw)
    if age_match:
        entities["age"] = int(age_match.group(1))

    return entities


def rag_agent_call(x_parsed: Dict[str, Any], collection) -> List[Dict[str, Any]]:
    """
    Retrieve top-k guideline chunks from Chroma based on parsed entities.
    """
    query_terms: List[str] = []

    for key in ("diagnoses", "medications"):
        for item in x_parsed.get(key, []):
            query_terms.append(str(item))

    if not query_terms:
        query_terms.append(x_parsed.get("text", ""))

    query = "; ".join(query_terms)

    results = collection.query(
        query_texts=[query],
        n_results=5,
    )

    retrieved: List[Dict[str, Any]] = []
    for i in range(len(results.get("ids", [[]])[0])):
        retrieved.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            }
        )
    return retrieved


def _normalize_gt_keys(gt_contradictions: str) -> List[str]:
    """
    Normalize ground-truth contradiction keys into a list.
    """
    value = str(gt_contradictions or "").strip()
    if not value:
        return []

    try:
        obj = json.loads(value)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, str):
            value = obj
    except json.JSONDecodeError:
        pass

    return [p.strip() for p in value.split(",") if p.strip()]


def audit_agent_call(
    x_parsed: Dict[str, Any],
    _retrieved: List[Dict[str, Any]],
    gt_contradictions: str,
    regime: str = "baseline_tau_old",
) -> Dict[str, Any]:
    """
    Rule-based clinical auditor with regime-aware thresholds.

    Checks multiple contradiction types:
    1. metformin_egfr_below_threshold: metformin used below regime-specific eGFR limit
    2. metformin_ckd_stage4: metformin used in CKD stage 4 (always contraindicated)

    Returns:
    - issues: human-readable messages
    - safe: bool (True if no active contradictions detected)
    - normalized_keys: list of canonical contradiction ids (for FCCR)
    """
    issues: List[str] = []
    normalized_keys: List[str] = []

    meds = [m.lower() for m in x_parsed.get("medications", [])]
    diagnoses = [d.lower() for d in x_parsed.get("diagnoses", [])]
    egfr = x_parsed.get("egfr")
    uses_metformin = any("metformin" in m for m in meds)

    if uses_metformin:
        # Rule 1: eGFR threshold (regime-dependent)
        threshold = REGIME_EGFR_THRESHOLDS.get(regime, 30)
        if egfr is not None and egfr < threshold:
            issues.append(
                f"Metformin contraindicated: eGFR {egfr:.1f} below "
                f"threshold {threshold} mL/min for regime {regime}."
            )
            normalized_keys.append("metformin_egfr_below_threshold")

        # Rule 2: CKD stage 4 (always contraindicated regardless of eGFR)
        has_ckd4 = any("ckd_stage4" in d for d in diagnoses)
        if has_ckd4:
            issues.append(
                "Metformin contraindicated in CKD stage 4 — "
                "risk of lactic acidosis."
            )
            normalized_keys.append("metformin_ckd_stage4")

    # Deduplicate
    normalized_keys = list(dict.fromkeys(normalized_keys))

    gt_keys = set(_normalize_gt_keys(gt_contradictions))
    detected_keys = set(normalized_keys)

    # Safe if no ground-truth contradictions exist, or all are detected
    safe = len(gt_keys) == 0 or detected_keys.issuperset(gt_keys)

    return {
        "issues": issues,
        "safe": safe,
        "normalized_keys": normalized_keys,
    }


def summary_agent_call(
    x_parsed: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    audit: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a final clinical summary grounded in retrieved guidelines
    and optionally incorporating audit feedback, via Ollama.
    """
    guideline_snippets = "\n\n".join(f"- {item['text']}" for item in retrieved[:3])

    audit_text = ""
    if audit and audit.get("issues"):
        bullet_issues = "\n".join(f"- {msg}" for msg in audit["issues"])
        audit_text = f"\n\nSafety issues detected:\n{bullet_issues}\n"

    prompt = (
        "You are a clinical summarization assistant.\n\n"
        f"Patient entities (JSON-like):\n{x_parsed}\n\n"
        f"Relevant guideline context:\n{guideline_snippets}\n\n"
        "Write a concise, factual summary of the patient's status and "
        "treatment plan.\n"
        "Explicitly respect any contraindications implied by the guidelines."
        f"\n{audit_text}"
    )
    return llm_complete(prompt)


def confidence_extraction_call(y_final: str, regime: str) -> float:
    """
    Extract verbalized confidence from the LLM about guideline alignment.

    Returns a float 0.0-1.0. Falls back to 0.5 if parsing fails.
    """
    epoch_year = "2026" if "tau_new" in regime else "2025"
    prompt = (
        f"You just generated this clinical summary:\n{y_final}\n\n"
        f"On a scale of 0.0 to 1.0, how confident are you that every "
        f"recommendation in this summary aligns with {epoch_year} clinical "
        f"guidelines? Return ONLY the number, nothing else."
    )
    raw = llm_complete(prompt).strip()
    try:
        val = float(raw.split()[0].strip(".,;:"))
        return max(0.0, min(1.0, val))
    except (ValueError, IndexError):
        return 0.5


# ---------------------------------------------------------------------
# Pipeline modes
# ---------------------------------------------------------------------


def single_agent_mode(x_raw: str, regime: str) -> Tuple[GlobalState, Dict[str, Any], float]:
    """
    Single-agent baseline: just call the LLM on the raw text
    (no RAG, no audit).
    """
    summary_prompt = (
        "You are a clinical summarization assistant.\n\n"
        f"Input clinical note:\n{x_raw}\n\n"
        "Write a concise, factual summary of the patient's status and "
        "treatment plan.\n"
        "Do not fabricate lab values or medications not present in the note."
    )
    y_final = llm_complete(summary_prompt)
    confidence = confidence_extraction_call(y_final, regime)

    state = GlobalState(
        x_raw=x_raw,
        x_parsed=None,
        retrieved=None,
        audit=None,
        y_final=y_final,
    )
    audit_meta: Dict[str, Any] = {
        "issues": [],
        "safe": True,
        "normalized_keys": [],
    }
    return state, audit_meta, confidence


def linear_multi_agent_mode(
    x_raw: str,
    gt_contradictions: str,
    collection,
    regime: str,
) -> Tuple[GlobalState, Dict[str, Any], float]:
    """
    Linear chain: parse -> RAG -> summary.
    Audit is used only for evaluation metadata, not routing.
    """
    state = GlobalState(x_raw=x_raw)

    x_parsed = parse_agent_call(x_raw)
    state.x_parsed = x_parsed

    retrieved = rag_agent_call(x_parsed, collection=collection)
    state.retrieved = retrieved

    y_final = summary_agent_call(x_parsed, retrieved, audit=None)
    state.y_final = y_final

    confidence = confidence_extraction_call(y_final, regime)

    audit_meta = audit_agent_call(x_parsed, retrieved, gt_contradictions, regime=regime)
    state.audit = audit_meta

    return state, audit_meta, confidence


def safety_graph_mode(
    x_raw: str,
    gt_contradictions: str,
    collection,
    regime: str,
    max_loops: int = 4,
) -> Tuple[GlobalState, Dict[str, Any], float]:
    """
    Stateful safety graph: parse -> RAG -> audit -> (loop) -> summary.
    """
    state = GlobalState(x_raw=x_raw)
    loop_count = 0
    last_audit: Optional[Dict[str, Any]] = None

    while True:
        if state.x_parsed is None or loop_count > 0:
            x_parsed = parse_agent_call(state.x_raw)
            state.x_parsed = x_parsed
        else:
            x_parsed = state.x_parsed

        retrieved = rag_agent_call(x_parsed, collection=collection)
        state.retrieved = retrieved

        audit_meta = audit_agent_call(
            x_parsed, retrieved, gt_contradictions, regime=regime
        )
        state.audit = audit_meta
        last_audit = audit_meta

        safe = bool(audit_meta.get("safe", False))
        loop_count += 1

        if safe or loop_count >= max_loops:
            break

    y_final = summary_agent_call(state.x_parsed, state.retrieved, audit=state.audit)
    state.y_final = y_final

    confidence = confidence_extraction_call(y_final, regime)

    if last_audit is None:
        last_audit = {"issues": [], "safe": True, "normalized_keys": []}

    return state, last_audit, confidence


# ---------------------------------------------------------------------
# Run helpers / CLI
# ---------------------------------------------------------------------


def run_once(
    row: pd.Series,
    regime: str,
    mode: str,
    collection,
) -> RunRecord:
    x_raw = row["Xraw"]
    y_star = row["y_star"]
    gt_contradictions = row.get("gtcontradictions", "")
    epoch_aligned = row.get("epoch_aligned", True)

    start = time.time()

    if mode == "single":
        state, audit_meta, confidence = single_agent_mode(x_raw, regime=regime)
    elif mode == "linear":
        state, audit_meta, confidence = linear_multi_agent_mode(
            x_raw=x_raw,
            gt_contradictions=gt_contradictions,
            collection=collection,
            regime=regime,
        )
    elif mode == "graph":
        state, audit_meta, confidence = safety_graph_mode(
            x_raw=x_raw,
            gt_contradictions=gt_contradictions,
            collection=collection,
            regime=regime,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    latency = time.time() - start
    condition = f"{mode}_{regime}"

    return RunRecord(
        patientid=row["patientid"],
        timeepoch=row["timeepoch"],
        y_final=state.y_final or "",
        y_star=y_star,
        gtcontradictions=str(gt_contradictions),
        auditissues=json.dumps(audit_meta, ensure_ascii=False),
        confidence=confidence,
        epoch_aligned=epoch_aligned,
        latency=latency,
        condition=condition,
    )


def run_condition(
    patients_path: Path,
    regime: str,
    mode: str,
    n: int,
    seed: int = 42,
) -> Path:
    if regime not in VALID_REGIMES:
        raise ValueError(f"regime must be one of {list(VALID_REGIMES.keys())}")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")

    df = load_patients(patients_path)
    if 0 < n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    # Prepare patient rows with regime-dependent ground-truth contradictions
    df_prepared = df.apply(prepare_patient_row, axis=1, regime=regime)

    collection_name = VALID_REGIMES[regime]
    collection = get_collection_for_regime(collection_name)

    out_name = f"results_{mode}_{regime}.csv"
    out_path = RESULTS_DIR / out_name

    # Checkpoint/resume: load already-processed patient IDs
    processed_ids: set = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        processed_ids = set(existing["patientid"].tolist())
        print(f"Resuming: {len(processed_ids)} patients already processed.")

    # Open in append mode for incremental writing
    write_header = not out_path.exists() or len(processed_ids) == 0
    import csv

    with open(out_path, "a", newline="") as f:
        writer = None
        for idx, row in df_prepared.iterrows():
            pid = row["patientid"]
            if pid in processed_ids:
                continue

            rec = run_once(row, regime=regime, mode=mode, collection=collection)
            rec_dict = rec.__dict__

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=rec_dict.keys())
                if write_header:
                    writer.writeheader()
            writer.writerow(rec_dict)
            f.flush()

            processed_ids.add(pid)

    print(f"Saved {out_path} ({len(processed_ids)} patients)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Run clinical agentic pipelines under shift regimes."
    )
    parser.add_argument(
        "--patients",
        type=str,
        default=str(DEFAULT_PATIENTS_PATH),
        help="Path to patients.csv",
    )
    parser.add_argument(
        "--regime",
        type=str,
        choices=list(VALID_REGIMES.keys()),
        help="Shift regime to simulate (omit for --run-all)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=VALID_MODES,
        help="Pipeline architecture mode (omit for --run-all)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Number of patients to run (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all mode x regime combinations sequentially",
    )
    args = parser.parse_args()

    if args.run_all:
        for regime in VALID_REGIMES:
            for mode in VALID_MODES:
                print(f"\n{'='*60}")
                print(f"Running: mode={mode}, regime={regime}, n={args.n}")
                print(f"{'='*60}")
                run_condition(
                    patients_path=Path(args.patients),
                    regime=regime,
                    mode=mode,
                    n=args.n,
                    seed=args.seed,
                )
    else:
        if not args.regime or not args.mode:
            parser.error("--regime and --mode are required unless --run-all is set")
        run_condition(
            patients_path=Path(args.patients),
            regime=args.regime,
            mode=args.mode,
            n=args.n,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
