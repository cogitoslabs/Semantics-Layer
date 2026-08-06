"""
utils/model_tracer.py — Decorator for logging model inputs, outputs, and generation parameters to CSV
"""

import csv
import functools
import inspect
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

logger = logging.getLogger("dapt.model_tracer")

CSV_HEADERS = [
    "Eval #",
    "Eval Category",
    "Eval Seq #",
    "Timestamp",
    "Function",
    "Prompt",
    "Output",
    "Parameters",
]


def _is_tracing_enabled() -> bool:
    """
    Check if model tracing is enabled via environment variable MODEL_TRACING
    or configuration setting.
    """
    val = os.getenv("MODEL_TRACING", "False").strip().lower()
    return val in ("true", "1", "t", "yes", "on")


def _get_trace_file_path() -> Path:
    """Get target CSV path for model tracing."""
    file_str = os.getenv("MODEL_TRACE_FILE", "logs/dapt_model_traces.csv")
    return Path(file_str)


def _format_cell(val: Any) -> str:
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    return str(val)


def _log_trace_to_csv(
    filepath: Path,
    function_name: str,
    prompt_data: Any,
    output_data: Any,
    params_dict: Dict[str, Any],
    eval_num: Any = "",
    eval_category: str = "",
    eval_seq_num: Any = None,
    eval_seq_start: int = 1,
) -> None:
    """Append model generation trace record(s) to the CSV file, unrolling batch items."""
    try:
        os.makedirs(filepath.parent, exist_ok=True)
        file_exists = filepath.exists() and filepath.stat().st_size > 0
        timestamp = datetime.now(timezone.utc).isoformat()
        params_str = json.dumps(params_dict)

        eval_num_str = str(eval_num) if eval_num is not None else ""
        eval_cat_str = str(eval_category) if eval_category is not None else ""

        is_batch = (
            isinstance(prompt_data, list)
            and isinstance(output_data, list)
            and len(prompt_data) == len(output_data)
        )

        rows = []
        if is_batch:
            start_seq = int(eval_seq_start) if eval_seq_start is not None else 1
            for i, (p, o) in enumerate(zip(prompt_data, output_data)):
                seq_val = start_seq + i
                rows.append({
                    "Eval #": eval_num_str,
                    "Eval Category": eval_cat_str,
                    "Eval Seq #": seq_val,
                    "Timestamp": timestamp,
                    "Function": function_name,
                    "Prompt": _format_cell(p),
                    "Output": _format_cell(o),
                    "Parameters": params_str,
                })
        else:
            seq_val = eval_seq_num if (eval_seq_num is not None and eval_seq_num != "") else 1
            rows.append({
                "Eval #": eval_num_str,
                "Eval Category": eval_cat_str,
                "Eval Seq #": seq_val,
                "Timestamp": timestamp,
                "Function": function_name,
                "Prompt": _format_cell(prompt_data),
                "Output": _format_cell(output_data),
                "Parameters": params_str,
            })

        with open(filepath, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        logger.warning(f"Failed to log model trace to {filepath}: {e}")


def model_trace(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for model generation / inference functions.
    When MODEL_TRACING is enabled (env var MODEL_TRACING=True or config),
    logs the complete prompt input, model output, and generation hyperparams to CSV.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)

        if not _is_tracing_enabled():
            return result

        try:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            all_params = bound_args.arguments

            prompt_val = all_params.get("prompts", all_params.get("prompt", ""))
            eval_num_val = all_params.get("eval_num", "")
            eval_cat_val = all_params.get("eval_category", "")
            eval_seq_val = all_params.get("eval_seq_num", None)
            eval_seq_start_val = all_params.get("eval_seq_start", 1)

            # Exclude non-serializable objects and trace meta-fields from Parameters column
            gen_params = {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in all_params.items()
                if k not in (
                    "model",
                    "tokenizer",
                    "prompt",
                    "prompts",
                    "self",
                    "cls",
                    "eval_num",
                    "eval_category",
                    "eval_seq_num",
                    "eval_seq_start",
                )
            }

            trace_path = _get_trace_file_path()
            _log_trace_to_csv(
                filepath=trace_path,
                function_name=fn.__name__,
                prompt_data=prompt_val,
                output_data=result,
                params_dict=gen_params,
                eval_num=eval_num_val,
                eval_category=eval_cat_val,
                eval_seq_num=eval_seq_val,
                eval_seq_start=eval_seq_start_val,
            )
        except Exception as e:
            logger.warning(f"Error during model tracing execution: {e}")

        return result

    return wrapper
