"""
lib/utils/trace_logger.py — Structured Per-Probe Evaluation Trace Logger and Cross-Checkpoint Comparator

Handles saving, listing, loading, and diffing per-probe evaluation traces in CSV format
under partitioned directories:
    logs/traces/
    ├── cloze/
    ├── qa/
    └── concept/
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from lib.utils.logger import get_logger

logger = get_logger("lib.utils.trace_logger")

STANDARD_TRACE_COLUMNS = [
    "eval_num",
    "seq_num",
    "prompt",
    "target_answer",
    "model_output",
    "matching_score",
    "result",
    "category",
    "checkpoint_name",
    "timestamp",
]

CATEGORY_ALIASES = {
    "cloze": "cloze",
    "cloze probe": "cloze",
    "qa": "qa",
    "qa probe": "qa",
    "concept": "concept",
    "concept probe": "concept",
}


def normalize_category_name(category: str) -> str:
    """Normalize probe category string to standard directory name (cloze, qa, concept)."""
    norm = category.strip().lower()
    return CATEGORY_ALIASES.get(norm, norm.replace(" ", "_"))


def format_file_label(file_path: Union[str, Path]) -> str:
    """Format file path into human readable label e.g., '2026-08-14 11:45:00 (eval_0001)'."""
    p = Path(file_path)
    stem = p.stem  # e.g., '20260814_114500_eval_0001'
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0].isdigit() and len(parts[0]) == 8:
        d_str = parts[0]
        t_str = parts[1]
        eval_part = "_".join(parts[2:])
        formatted_date = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}"
        formatted_time = f"{t_str[:2]}:{t_str[2:4]}:{t_str[4:6]}" if len(t_str) >= 6 else t_str
        return f"{formatted_date} {formatted_time} ({eval_part})"
    return p.name


def parse_eval_num_from_filename(filename: Union[str, Path]) -> int:
    """Extract eval number integer from filename for intelligent baseline sorting."""
    stem = Path(filename).stem
    if "eval_" in stem:
        try:
            val = stem.split("eval_")[-1]
            return int(val)
        except ValueError:
            pass
    return 999999



def normalize_trace_record(
    record: Dict[str, Any],
    eval_num: Union[int, str],
    category: str,
    seq_idx: int = 1,
    checkpoint_name: str = "",
    timestamp_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize raw evaluation probe trace dictionaries into standard schema."""
    ts = timestamp_str or datetime.now(timezone.utc).isoformat()
    
    # Extract sequence number
    seq_num = record.get("seq_num")
    if seq_num is None:
        seq_num = record.get("Eval Seq #", seq_idx)
    try:
        seq_num = int(seq_num)
    except (ValueError, TypeError):
        seq_num = seq_idx

    # Extract prompt
    prompt = record.get("prompt")
    if prompt is None:
        prompt = record.get("question")
    if prompt is None and "Eval" in record:
        try:
            eval_data = json.loads(record["Eval"]) if isinstance(record["Eval"], str) else record["Eval"]
            if isinstance(eval_data, dict):
                prompt = eval_data.get("prompt") or eval_data.get("question")
        except Exception:
            prompt = str(record.get("Eval", ""))
    prompt = str(prompt or "")

    # Extract target answer
    target_answer = record.get("target_answer")
    if target_answer is None:
        target_answer = record.get("target_term")
    if target_answer is None:
        target_answer = record.get("expected_text")
    if target_answer is None:
        target_answer = record.get("reference")
    if target_answer is None and "Eval" in record:
        try:
            eval_data = json.loads(record["Eval"]) if isinstance(record["Eval"], str) else record["Eval"]
            if isinstance(eval_data, dict):
                target_answer = (
                    eval_data.get("target")
                    or eval_data.get("target_term")
                    or eval_data.get("expected_text")
                    or eval_data.get("reference")
                )
        except Exception:
            pass
    target_answer = str(target_answer or "")

    # Extract model output
    model_output = record.get("model_output")
    if model_output is None:
        model_output = record.get("Generated Answer by the model")
    if model_output is None:
        model_output = record.get("predicted_text")
    if model_output is None:
        model_output = record.get("generated")
    if model_output is None:
        model_output = record.get("predicted_choice")
    model_output = str(model_output or "")

    # Extract matching score
    matching_score = record.get("matching_score")
    if matching_score is None:
        matching_score = record.get("Matching Score")
    if matching_score is None:
        matching_score = record.get("score", 0.0)
    try:
        matching_score = round(float(matching_score), 4)
    except (ValueError, TypeError):
        matching_score = 0.0

    # Extract result
    result = record.get("result")
    if result is None:
        result = record.get("Result", "Fail")
    result = str(result)

    # Extract sub-category / cluster
    sub_category = record.get("category")
    if sub_category is None:
        sub_category = record.get("cluster")
    if sub_category is None:
        sub_category = record.get("Eval Category", category)
    sub_category = str(sub_category or "")

    ckpt = str(checkpoint_name or record.get("checkpoint_name") or f"eval_{eval_num}")

    return {
        "eval_num": str(eval_num),
        "seq_num": seq_num,
        "prompt": prompt,
        "target_answer": target_answer,
        "model_output": model_output,
        "matching_score": matching_score,
        "result": result,
        "category": sub_category,
        "checkpoint_name": ckpt,
        "timestamp": ts,
    }


def save_probe_traces_csv(
    category: str,
    eval_num: Union[int, str],
    traces: List[Dict[str, Any]],
    checkpoint_name: str = "",
    base_dir: Union[str, Path] = "logs/traces",
    run_id: Optional[str] = None,
    timestamp_str: Optional[str] = None,
) -> Path:
    """
    Save list of evaluation traces into a dedicated per-probe timestamped CSV file.
    
    Path format:
      If run_id is provided: {base_dir}/{run_id}/{normalized_category}/eval_{eval_num:04d}.csv
      Otherwise: {base_dir}/{normalized_category}/{timestamp}_eval_{eval_num:04d}.csv
    """
    norm_category = normalize_category_name(category)
    now = datetime.now(timezone.utc)
    ts_file = now.strftime("%Y%m%d_%H%M%S")
    ts_iso = timestamp_str or now.isoformat()

    eval_str = f"{int(eval_num):04d}" if str(eval_num).isdigit() else str(eval_num)

    if run_id:
        target_dir = Path(base_dir) / str(run_id).strip() / norm_category
        filename = f"eval_{eval_str}.csv"
    else:
        target_dir = Path(base_dir) / norm_category
        filename = f"{ts_file}_eval_{eval_str}.csv"

    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / filename

    normalized_records = [
        normalize_trace_record(
            record=trace,
            eval_num=eval_num,
            category=norm_category,
            seq_idx=idx + 1,
            checkpoint_name=checkpoint_name,
            timestamp_str=ts_iso,
        )
        for idx, trace in enumerate(traces)
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_TRACE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in normalized_records:
            writer.writerow(rec)

    logger.info(f"Saved {len(normalized_records)} {norm_category} trace rows to {output_path}")
    return output_path


def list_trace_categories(base_dir: Union[str, Path] = "logs/traces") -> List[str]:
    """List all available probe trace categories having files, or standard probe categories."""
    base_path = Path(base_dir)
    default_categories = ["cloze", "qa", "concept"]
    if not base_path.exists():
        return default_categories
    
    found = [d.name for d in base_path.iterdir() if d.is_dir()]
    # Ensure standard order or add any found ones
    categories = [c for c in default_categories if c in found]
    for d in found:
        if d not in categories:
            categories.append(d)
    return categories or default_categories


def list_trace_files(
    category: str,
    base_dir: Union[str, Path] = "logs/traces",
) -> List[Path]:
    """Scan directory and return list of trace CSV files for a probe category sorted newest first."""
    norm_category = normalize_category_name(category)
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    # Check direct category folder and nested run_id/category folders
    nested_files = list(base_path.rglob(f"{norm_category}/*.csv"))
    flat_files = [f for f in base_path.rglob("*.csv") if f.parent.name == norm_category or f"{norm_category}_" in f.name or f"_{norm_category}." in f.name]
    all_files = list({str(f): f for f in (nested_files + flat_files)}.values())
    all_files.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return all_files


def load_trace_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a trace CSV into a pandas DataFrame with guaranteed standard columns."""
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=STANDARD_TRACE_COLUMNS)
    
    df = pd.read_csv(p, encoding="utf-8", dtype=str)
    # Ensure all standard columns exist
    for col in STANDARD_TRACE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    
    # Cast numerical columns safely
    df["seq_num"] = pd.to_numeric(df["seq_num"], errors="coerce").fillna(0).astype(int)
    df["matching_score"] = pd.to_numeric(df["matching_score"], errors="coerce").fillna(0.0).astype(float)
    return df


def compute_trace_diff(
    base_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare baseline traces against target checkpoint traces on seq_num.
    
    Returns a unified DataFrame containing baseline fields (suffix `_base`),
    target fields (suffix `_target`), and comparison indicators:
      - `is_output_changed`: bool
      - `is_result_changed`: bool
      - `delta_status`: "Improved" (Fail -> Pass), "Regressed" (Pass -> Fail),
                       "Unchanged Pass", "Unchanged Fail"
    """
    if base_df.empty and target_df.empty:
        return pd.DataFrame()
    
    if base_df.empty:
        # Fallback when no baseline is available (Single Checkpoint View)
        res = target_df.copy()
        res["prompt_display"] = res["prompt"] if "prompt" in res.columns else ""
        res["target_answer_display"] = res["target_answer"] if "target_answer" in res.columns else ""
        res["category_display"] = res["category"] if "category" in res.columns else ""
        res["model_output_base"] = "(No baseline available)"
        res["matching_score_base"] = 0.0
        res["result_base"] = "N/A"
        res["model_output_target"] = res["model_output"] if "model_output" in res.columns else ""
        res["matching_score_target"] = res["matching_score"] if "matching_score" in res.columns else 0.0
        res["result_target"] = res["result"] if "result" in res.columns else "N/A"
        res["error_category_target"] = res["error_category"] if "error_category" in res.columns else ""
        res["notes_target"] = res["notes"] if "notes" in res.columns else ""
        res["is_output_changed"] = True
        res["is_result_changed"] = False
        res["delta_status"] = "Single Checkpoint"
        return res

    if target_df.empty:
        res = base_df.copy()
        res["prompt_display"] = res["prompt"] if "prompt" in res.columns else ""
        res["target_answer_display"] = res["target_answer"] if "target_answer" in res.columns else ""
        res["category_display"] = res["category"] if "category" in res.columns else ""
        res["model_output_base"] = res["model_output"] if "model_output" in res.columns else ""
        res["matching_score_base"] = res["matching_score"] if "matching_score" in res.columns else 0.0
        res["result_base"] = res["result"] if "result" in res.columns else "N/A"
        res["model_output_target"] = "(No checkpoint data)"
        res["matching_score_target"] = 0.0
        res["result_target"] = "N/A"
        res["error_category_target"] = res["error_category"] if "error_category" in res.columns else ""
        res["notes_target"] = res["notes"] if "notes" in res.columns else ""
        res["is_output_changed"] = False
        res["is_result_changed"] = False
        res["delta_status"] = "Baseline Only"
        return res

    # Outer merge on seq_num
    merged = pd.merge(
        base_df,
        target_df,
        on="seq_num",
        how="outer",
        suffixes=("_base", "_target"),
    )

    # Clean text columns
    for suffix in ["_base", "_target"]:
        for col in ["prompt", "target_answer", "model_output", "result", "category"]:
            c = f"{col}{suffix}"
            if c in merged.columns:
                merged[c] = merged[c].fillna("").astype(str)

    merged["matching_score_base"] = pd.to_numeric(merged.get("matching_score_base"), errors="coerce").fillna(0.0)
    merged["matching_score_target"] = pd.to_numeric(merged.get("matching_score_target"), errors="coerce").fillna(0.0)

    merged["prompt_display"] = merged["prompt_target"].where(merged["prompt_target"] != "", merged["prompt_base"])
    merged["target_answer_display"] = merged["target_answer_target"].where(
        merged["target_answer_target"] != "", merged["target_answer_base"]
    )
    merged["category_display"] = merged["category_target"].where(
        merged["category_target"] != "", merged["category_base"]
    )

    base_out = merged["model_output_base"].str.strip()
    tgt_out = merged["model_output_target"].str.strip()
    merged["is_output_changed"] = base_out != tgt_out

    base_res = merged["result_base"].str.strip().str.capitalize()
    tgt_res = merged["result_target"].str.strip().str.capitalize()
    merged["is_result_changed"] = base_res != tgt_res

    def determine_delta(row):
        b_res = str(row.get("result_base", "")).strip().capitalize()
        t_res = str(row.get("result_target", "")).strip().capitalize()
        if b_res == "Fail" and t_res == "Pass":
            return "Improved"
        elif b_res == "Pass" and t_res == "Fail":
            return "Regressed"
        elif t_res == "Pass":
            return "Unchanged Pass"
        elif t_res == "Fail":
            return "Unchanged Fail"
        else:
            return "Unchanged"

    merged["delta_status"] = merged.apply(determine_delta, axis=1)
    merged = merged.sort_values(by="seq_num").reset_index(drop=True)
    return merged
