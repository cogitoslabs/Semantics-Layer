import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from lib.utils import PipelineConfig

logger = logging.getLogger(__name__)


def run_benchmark_reporting(
    cfg: PipelineConfig,
    records: List[Dict[str, Any]],
    expected_teacher_count: int,
    expected_cluster_count: int
) -> Dict[str, Any]:
    """
    Aggregates per-sample BenchmarkRecord records into per-teacher × cluster scores.
    Enforces validation gates, generates scores.jsonl and benchmark_manifest.json.
    """
    logger.info("Initializing benchmark reporter...")
    
    # 1. Group records by (teacher_model, cluster_label)
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in records:
        key = (r["teacher_model"], r["cluster_label"])
        grouped.setdefault(key, []).append(r)
        
    # Calculate scores per teacher × cluster
    cluster_scores = []
    
    # Track statistics for validation gates
    warnings: List[str] = []
    
    teacher_null_counts: Dict[str, int] = {}
    teacher_total_samples: Dict[str, int] = {}
    teacher_hallucinations: Dict[str, List[float]] = {}
    
    cluster_sample_sizes: Dict[str, int] = {}
    
    for (teacher, cluster_label), sample_records in sorted(grouped.items()):
        total_samples = len(sample_records)
        cluster_id = sample_records[0]["cluster_id"]
        
        # Calculate aggregates
        acc_scores = [r["answer_accuracy"] for r in sample_records]
        res_scores = [r["reasoning_quality"] for r in sample_records if r["reasoning_quality"] is not None]
        
        # Count null reasoning_quality
        null_count = sum(1 for r in sample_records if r["reasoning_quality"] is None)
        teacher_null_counts[teacher] = teacher_null_counts.get(teacher, 0) + null_count
        teacher_total_samples[teacher] = teacher_total_samples.get(teacher, 0) + total_samples
        
        # Citation accuracy (exclude no-retrieval)
        cit_precisions = [r["citation_precision"] for r in sample_records if not r["no_retrieval"] and r["citation_precision"] is not None]
        cit_recalls = [r["citation_recall"] for r in sample_records if not r["no_retrieval"] and r["citation_recall"] is not None]
        cit_accs = [r["citation_accuracy"] for r in sample_records if not r["no_retrieval"] and r["citation_accuracy"] is not None]
        
        # Hallucination rate
        hal_rates = [r["hallucination_rate"] for r in sample_records]
        for hr in hal_rates:
            teacher_hallucinations.setdefault(teacher, []).append(hr)
            
        no_ret_count = sum(1 for r in sample_records if r["no_retrieval"])
        
        # Track sample size per cluster (shared across teachers)
        cluster_sample_sizes[cluster_label] = total_samples
        
        # Aggregate means
        mean_acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0.0
        mean_res = sum(res_scores) / len(res_scores) if res_scores else 0.0
        mean_cit_prec = sum(cit_precisions) / len(cit_precisions) if cit_precisions else None
        mean_cit_rec = sum(cit_recalls) / len(cit_recalls) if cit_recalls else None
        mean_cit_acc = sum(cit_accs) / len(cit_accs) if cit_accs else None
        mean_hal = sum(hal_rates) / len(hal_rates) if hal_rates else 0.0
        no_ret_frac = no_ret_count / total_samples if total_samples > 0 else 0.0
        
        score_record = {
            "teacher_model": teacher,
            "cluster_label": cluster_label,
            "cluster_id": int(cluster_id) if cluster_id is not None else -1,
            "eval_sample_size": total_samples,
            "answer_accuracy": float(f"{mean_acc:.4f}"),
            "reasoning_quality": float(f"{mean_res:.4f}"),
            "citation_precision": float(f"{mean_cit_prec:.4f}") if mean_cit_prec is not None else None,
            "citation_recall": float(f"{mean_cit_rec:.4f}") if mean_cit_rec is not None else None,
            "citation_accuracy": float(f"{mean_cit_acc:.4f}") if mean_cit_acc is not None else None,
            "hallucination_rate": float(f"{mean_hal:.4f}"),
            "no_retrieval_fraction": float(f"{no_ret_frac:.4f}")
        }
        cluster_scores.append(score_record)
        
    # 2. Write data/benchmarking/scores.jsonl
    scores_path = Path(cfg.benchmarking.scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing per-teacher × cluster scores to {scores_path}...")
    with open(scores_path, "w", encoding="utf-8") as f:
        for score in cluster_scores:
            f.write(json.dumps(score) + "\n")
            
    # 3. Calculate per-teacher aggregates for manifest
    per_teacher_aggregate = {}
    for teacher in cfg.benchmarking.candidate_teachers:
        teacher_records = [sc for sc in cluster_scores if sc["teacher_model"] == teacher]
        if not teacher_records:
            continue
            
        t_accs = [sc["answer_accuracy"] for sc in teacher_records]
        t_res = [sc["reasoning_quality"] for sc in teacher_records]
        t_cit = [sc["citation_accuracy"] for sc in teacher_records if sc["citation_accuracy"] is not None]
        t_hal = [sc["hallucination_rate"] for sc in teacher_records]
        
        per_teacher_aggregate[teacher] = {
            "mean_answer_accuracy": float(f"{sum(t_accs) / len(t_accs):.4f}") if t_accs else 0.0,
            "mean_reasoning_quality": float(f"{sum(t_res) / len(t_res):.4f}") if t_res else 0.0,
            "mean_citation_accuracy": float(f"{sum(t_cit) / len(t_cit):.4f}") if t_cit else None,
            "mean_hallucination_rate": float(f"{sum(t_hal) / len(t_hal):.4f}") if t_hal else 0.0
        }
        
    # 4. Enforce validation gates
    actual_records_written = len(cluster_scores)
    expected_records = expected_teacher_count * expected_cluster_count
    
    # Hard Fail check
    if actual_records_written != expected_records:
        raise ValueError(
            f"Validation Gate Hard Fail: Mismatch in score records written. "
            f"Expected {expected_records} (teachers: {expected_teacher_count} × clusters: {expected_cluster_count}), "
            f"but wrote {actual_records_written}."
        )
        
    # Check reasoning quality null rate
    for teacher, null_count in teacher_null_counts.items():
        total = teacher_total_samples.get(teacher, 0)
        null_rate = null_count / total if total > 0 else 0.0
        if null_rate > 0.05:
            warnings.append(
                f"reasoning_quality null rate ({null_rate:.2%}) is > 5% for teacher {teacher} "
                f"({null_count} nulls out of {total} samples)."
            )
            
    # Check mean hallucination rate
    for teacher, hr_list in teacher_hallucinations.items():
        mean_hr = sum(hr_list) / len(hr_list) if hr_list else 0.0
        if mean_hr > 0.50:
            warnings.append(
                f"Mean hallucination_rate ({mean_hr:.2%}) is > 50% for teacher {teacher}."
            )
            
    # Check eval sample count
    for cluster, size in cluster_sample_sizes.items():
        if size < cfg.benchmarking.min_eval_samples:
            warnings.append(
                f"Eval sample coverage for cluster {cluster} is {size}, which is below min_eval_samples={cfg.benchmarking.min_eval_samples}."
            )
            
    # Manifest status
    status = "complete"  # Since hard fail throws exception, if we reach here it's clean or has warnings only
    
    # Check if judge calibration has been run
    judge_calibrated = False
    cal_log_path = Path(cfg.benchmarking.calibration_log_path)
    inter_log_path = Path(cfg.benchmarking.inter_rater_log_path)
    if cfg.benchmarking.enable_calibration and cal_log_path.exists() and inter_log_path.exists():
        judge_calibrated = True
        
    manifest = {
        "status": status,
        "candidate_teachers": cfg.benchmarking.candidate_teachers,
        "total_clusters": expected_cluster_count,
        "eval_sample_size": cfg.benchmarking.eval_sample_size,
        "judge_model": cfg.benchmarking.judge_model_name,
        "judge_calibrated": judge_calibrated,
        "per_teacher_aggregate": per_teacher_aggregate,
        "warnings": warnings
    }
    
    manifest_path = Path(cfg.benchmarking.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing benchmark manifest to {manifest_path}...")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    return manifest
