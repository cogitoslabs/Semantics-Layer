"""
ui/trace_log.py — Streamlit Web Application for Evaluation Trace Log Inspection & Comparison

Allows interactive browsing and side-by-side comparison of evaluation probe completions
between baseline and fine-tuned checkpoints across Cloze, QA, and Concept probes.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import streamlit as st

from lib.utils.config import PipelineConfig
from lib.utils.trace_logger import (
    compute_trace_diff,
    format_file_label,
    list_trace_categories,
    list_trace_files,
    load_trace_file,
    normalize_category_name,
    parse_eval_num_from_filename,
)



def main():
    st.set_page_config(
        page_title="Evaluation Trace Logs",
        page_icon="🔍",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem !important;
                padding-bottom: 1.5rem !important;
            }
            .metric-card {
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 12px 16px;
                text-align: center;
            }
            .trace-header-card {
                background: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 16px;
            }
            .output-card {
                border-radius: 8px;
                padding: 16px;
                height: 100%;
            }
            .badge-pass {
                background-color: #d1e7dd;
                color: #0f5132;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: 600;
            }
            .badge-fail {
                background-color: #f8d7da;
                color: #842029;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: 600;
            }
            .badge-improved {
                background-color: #d1e7dd;
                color: #0f5132;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: 700;
            }
            .badge-regressed {
                background-color: #f8d7da;
                color: #842029;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: 700;
            }
            .badge-neutral {
                background-color: #e2e3e5;
                color: #41464b;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cfg = PipelineConfig()
    trace_base_dir = cfg.logging.log_dir / "traces"

    # Sidebar setup
    st.sidebar.title("Trace Log Browser")
    st.sidebar.markdown("Inspect and compare evaluation probe predictions.")

    # 1. Category selector
    categories = ["Cloze", "QA", "Concept"]
    selected_category = st.sidebar.selectbox("Evaluation Probe", categories, index=0)
    norm_cat = normalize_category_name(selected_category)

    # 2. Find available trace files
    trace_files = list_trace_files(norm_cat, base_dir=trace_base_dir)

    if not trace_files:
        st.sidebar.warning(f"No trace logs found for `{selected_category}` in `{trace_base_dir / norm_cat}`.")
        st.info(
            f"### No `{selected_category}` Traces Available\n\n"
            f"Traces are automatically saved during evaluation passes (`eval_runner.py`).\n\n"
            f"Expected trace directory: `{trace_base_dir / norm_cat}`"
        )
        return

    # Map files to readable labels
    file_map = {f: format_file_label(f) for f in trace_files}
    file_options = list(file_map.keys())

    # 3. Checkpoint Selector (Default to most recent)
    selected_target_file = st.sidebar.selectbox(
        "Selected Checkpoint Trace",
        file_options,
        format_func=lambda f: file_map[f],
        index=0,
        help="Select evaluation cycle to inspect (ordered newest to oldest).",
    )

    # 4. Baseline Selector (Default to earliest / eval_0)
    baseline_options = [None] + file_options
    
    # Auto-select earliest or eval_0
    default_base_idx = 0
    if len(file_options) > 1:
        # Find file with smallest eval number or oldest timestamp
        earliest_file = min(file_options, key=lambda f: (parse_eval_num_from_filename(f.name), f.stat().st_mtime))
        try:
            default_base_idx = baseline_options.index(earliest_file)
        except ValueError:
            default_base_idx = len(baseline_options) - 1

    def format_base_label(f):
        if f is None:
            return "(None - Single Checkpoint View)"
        return file_map.get(f, str(f))

    selected_base_file = st.sidebar.selectbox(
        "Base Model / Reference Trace",
        baseline_options,
        format_func=format_base_label,
        index=default_base_idx,
        help="Select baseline trace to compare against.",
    )

    st.sidebar.markdown("---")

    # 5. Filters
    show_changed_only = st.sidebar.checkbox(
        "Show only traces changed from base",
        value=False,
        help="Filter items where the selected checkpoint generated a different answer than the baseline.",
    )

    # Load DataFrames
    target_df = load_trace_file(selected_target_file) if selected_target_file else pd.DataFrame()
    base_df = load_trace_file(selected_base_file) if selected_base_file else pd.DataFrame()

    diff_df = compute_trace_diff(base_df, target_df)

    if diff_df.empty:
        st.warning("Selected trace file contains no data rows.")
        return

    # Subcategory filter if available
    available_subcats = sorted([c for c in diff_df["category_display"].unique() if str(c).strip()])
    if available_subcats:
        selected_subcat = st.sidebar.selectbox(
            "Filter Category / Cluster",
            ["All Categories"] + available_subcats,
        )
        if selected_subcat != "All Categories":
            diff_df = diff_df[diff_df["category_display"] == selected_subcat]

    # Search filter
    search_query = st.sidebar.text_input("Search Prompt or Answer", "").strip().lower()
    if search_query:
        diff_df = diff_df[
            diff_df["prompt_display"].str.lower().str.contains(search_query, na=False)
            | diff_df["target_answer_display"].str.lower().str.contains(search_query, na=False)
            | diff_df["model_output_target"].str.lower().str.contains(search_query, na=False)
        ]

    # Apply changed-only filter
    if show_changed_only:
        diff_df = diff_df[diff_df["is_output_changed"]]

    total_filtered = len(diff_df)

    # Header and Summary Metrics
    st.header(f"🔍 {selected_category} Probe Traces")

    # Top Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    total_items = len(target_df) if not target_df.empty else len(base_df)
    changed_count = int(diff_df["is_output_changed"].sum()) if not diff_df.empty else 0

    base_pass_count = int((base_df["result"].str.strip().str.capitalize() == "Pass").sum()) if not base_df.empty else 0
    base_pass_rate = (base_pass_count / len(base_df) * 100) if not base_df.empty else 0.0

    target_pass_count = int((target_df["result"].str.strip().str.capitalize() == "Pass").sum()) if not target_df.empty else 0
    target_pass_rate = (target_pass_count / len(target_df) * 100) if not target_df.empty else 0.0
    delta_rate = target_pass_rate - base_pass_rate

    m_col1.metric("Total Traces", total_items)
    m_col2.metric("Changed Completions", f"{changed_count} ({changed_count/max(1, total_items)*100:.1f}%)")
    m_col3.metric("Base Pass Rate", f"{base_pass_rate:.1f}%" if not base_df.empty else "N/A", f"{base_pass_count}/{len(base_df)}" if not base_df.empty else "")
    m_col4.metric(
        "Checkpoint Pass Rate",
        f"{target_pass_rate:.1f}%",
        f"{target_pass_count}/{len(target_df)}",
    )
    m_col5.metric(
        "Net Accuracy Delta",
        f"{delta_rate:+.1f}%" if not base_df.empty else "N/A",
        delta=f"{delta_rate:+.1f}%" if not base_df.empty else None,
    )

    st.markdown("---")

    if total_filtered == 0:
        st.info("No traces match the selected filters.")
        return

    # Item Navigation Controls
    if "trace_index" not in st.session_state:
        st.session_state["trace_index"] = 0

    # Ensure index in valid range
    if st.session_state["trace_index"] >= total_filtered:
        st.session_state["trace_index"] = max(0, total_filtered - 1)

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 4, 2])

    if nav_col1.button("⬅️ Previous", disabled=(st.session_state["trace_index"] == 0), use_container_width=True):
        st.session_state["trace_index"] = max(0, st.session_state["trace_index"] - 1)
        st.rerun()

    if nav_col2.button("Next ➡️", disabled=(st.session_state["trace_index"] >= total_filtered - 1), use_container_width=True):
        st.session_state["trace_index"] = min(total_filtered - 1, st.session_state["trace_index"] + 1)
        st.rerun()

    curr_idx = st.session_state["trace_index"]
    
    # Slider navigation
    selected_slider_idx = nav_col3.slider(
        "Jump to Item",
        min_value=1,
        max_value=total_filtered,
        value=curr_idx + 1,
        label_visibility="collapsed",
    )
    if selected_slider_idx - 1 != curr_idx:
        st.session_state["trace_index"] = selected_slider_idx - 1
        st.rerun()

    nav_col4.markdown(f"**Item {curr_idx + 1} of {total_filtered}**")

    # Current Item Record
    row = diff_df.iloc[curr_idx]
    seq_num = row.get("seq_num", curr_idx + 1)
    category_val = row.get("category_display", "")
    prompt_text = row.get("prompt_display", "")
    target_answer_text = row.get("target_answer_display", "")

    # Top Prompt Card
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 14px 18px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; color: #495057;">Sequence #{seq_num}</span>
                <span style="background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; color: #495057;">Category: {category_val or 'Standard'}</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #212529; margin-bottom: 6px;">Prompt:</div>
            <div style="background: #ffffff; border: 1px solid #ced4da; border-radius: 4px; padding: 10px; margin-bottom: 10px; font-family: monospace; white-space: pre-wrap;">{prompt_text}</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #495057;">Expected Target Answer:</div>
            <div style="background: #e8f5e9; border: 1px solid #c8e6c9; color: #1b5e20; border-radius: 4px; padding: 8px; font-family: monospace;">{target_answer_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status & Delta Banner
    delta_status = row.get("delta_status", "Unchanged")
    is_changed = row.get("is_output_changed", False)
    
    status_color = "#6c757d"
    status_label = "Unchanged"
    if delta_status == "Improved":
        status_color = "#198754"
        status_label = "🟢 Improved (Fail ➔ Pass)"
    elif delta_status == "Regressed":
        status_color = "#dc3545"
        status_label = "🔴 Regressed (Pass ➔ Fail)"
    elif delta_status == "Unchanged Pass":
        status_color = "#198754"
        status_label = "🟢 Maintained (Pass ➔ Pass)"
    elif delta_status == "Unchanged Fail":
        status_color = "#dc3545"
        status_label = "🔴 Maintained (Fail ➔ Fail)"

    output_diff_badge = "🔄 Output Changed" if is_changed else "⚪ Output Identical"

    st.markdown(
        f"""
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 14px;">
            <span style="font-weight: 700; color: {status_color}; font-size: 1.05rem;">{status_label}</span>
            <span style="background: #e2e3e5; color: #383d41; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">{output_diff_badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Side-by-Side Comparison Columns
    col_base, col_target = st.columns(2)

    # Left Column: Base Model Output
    with col_base:
        st.subheader("Base Model Output")
        base_out = str(row.get("model_output_base", "")).strip() or "(No output recorded)"
        base_score = row.get("matching_score_base", 0.0)
        base_res = str(row.get("result_base", "N/A")).strip().capitalize()
        
        b_badge_class = "badge-pass" if base_res == "Pass" else ("badge-fail" if base_res == "Fail" else "badge-neutral")
        
        st.markdown(
            f"""
            <div style="border: 1px solid #ced4da; border-radius: 6px; padding: 12px; background: #ffffff; min-height: 180px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span class="{b_badge_class}">{base_res}</span>
                    <span style="font-size: 0.9rem; color: #6c757d;">Score: <strong>{base_score:.4f}</strong></span>
                </div>
                <div style="font-family: monospace; white-space: pre-wrap; font-size: 0.95rem; color: #212529;">{base_out}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Right Column: Checkpoint Output
    with col_target:
        st.subheader("Selected Checkpoint Output")
        tgt_out = str(row.get("model_output_target", "")).strip() or "(No output recorded)"
        tgt_score = row.get("matching_score_target", 0.0)
        tgt_res = str(row.get("result_target", "N/A")).strip().capitalize()
        
        t_badge_class = "badge-pass" if tgt_res == "Pass" else ("badge-fail" if tgt_res == "Fail" else "badge-neutral")
        
        # Border highlight if improved/regressed
        border_color = "#198754" if delta_status == "Improved" else ("#dc3545" if delta_status == "Regressed" else "#0d6efd")

        st.markdown(
            f"""
            <div style="border: 2px solid {border_color}; border-radius: 6px; padding: 12px; background: #ffffff; min-height: 180px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span class="{t_badge_class}">{tgt_res}</span>
                    <span style="font-size: 0.9rem; color: #6c757d;">Score: <strong>{tgt_score:.4f}</strong></span>
                </div>
                <div style="font-family: monospace; white-space: pre-wrap; font-size: 0.95rem; color: #212529;">{tgt_out}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tabular Table View
    with st.expander("📋 View All Filtered Traces in Table"):
        display_cols = [
            "seq_num",
            "category_display",
            "prompt_display",
            "target_answer_display",
            "result_base",
            "result_target",
            "delta_status",
            "is_output_changed",
            "model_output_base",
            "model_output_target",
        ]
        available_cols = [c for c in display_cols if c in diff_df.columns]
        st.dataframe(diff_df[available_cols], use_container_width=True)


if __name__ == "__main__":
    main()
