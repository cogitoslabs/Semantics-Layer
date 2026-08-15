"""
ui/trace_log.py — Streamlit Web Application for Evaluation Trace Inspection & Error Categorization

Connects directly to the SQLite database (logs/traces.db) with automatic CSV sync.
Allows interactive browsing, cross-checkpoint diff comparison, dynamic error categorization,
and failure taxonomy tagging across Cloze, QA, and Concept probes.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import importlib
import pandas as pd
import streamlit as st

import lib.utils.trace_db
import lib.utils.trace_logger
importlib.reload(lib.utils.trace_db)
importlib.reload(lib.utils.trace_logger)

from lib.utils.config import PipelineConfig
from lib.utils.trace_db import (
    get_distinct_error_categories,
    init_trace_db,
    list_db_runs,
    list_db_checkpoints_for_run,
    list_db_runs_and_checkpoints,
    load_probe_traces_from_db,
    normalize_probe_name,
    sync_all_traces_to_db,
    update_trace_annotation,
)
from lib.utils.trace_logger import compute_trace_diff


def main():
    st.set_page_config(
        page_title="Evaluation Trace Logs",
        page_icon="🔍",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            header[data-testid="stHeader"] {
                background: transparent !important;
                height: 2.5rem !important;
            }
            .block-container {
                padding-top: 2.8rem !important;
                padding-bottom: 1.2rem !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.35rem !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.8rem !important;
                font-weight: 600;
            }
            [data-testid="stMetricDelta"] {
                font-size: 0.75rem !important;
            }
            [data-testid="stMetric"] {
                padding: 2px 4px !important;
            }
            hr {
                margin: 0.5rem 0 !important;
            }
            .badge-pass {
                background-color: #d1e7dd;
                color: #0f5132;
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.85rem;
            }
            .badge-fail {
                background-color: #f8d7da;
                color: #842029;
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.85rem;
            }
            .badge-improved {
                background-color: #d1e7dd;
                color: #0f5132;
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.85rem;
            }
            .badge-regressed {
                background-color: #f8d7da;
                color: #842029;
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.85rem;
            }
            .badge-neutral {
                background-color: #e2e3e5;
                color: #41464b;
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.85rem;
            }
            .badge-category {
                background-color: #ede9fe;
                color: #5b21b6;
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.85rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cfg = PipelineConfig()
    trace_base_dir = cfg.logging.log_dir / "traces"
    trace_db_path = cfg.logging.trace_db_path

    # Initialize SQLite database schema
    init_trace_db(trace_db_path)

    # Auto-sync trace CSV files to SQLite on startup
    if "db_synced" not in st.session_state:
        sync_all_traces_to_db(trace_base_dir, trace_db_path)
        st.session_state["db_synced"] = True

    # Sidebar setup
    st.sidebar.title("Trace Log Browser")
    st.sidebar.markdown("Inspect, compare, and categorize evaluation probe predictions.")

    # Manual Sync Button
    if st.sidebar.button("🔄 Sync Traces from Logs", help="Reload all trace CSVs into SQLite database"):
        counts = sync_all_traces_to_db(trace_base_dir, trace_db_path)
        total_synced = sum(counts.values())
        st.sidebar.success(f"Synced {total_synced:,} records across probes!")
        st.rerun()

    st.sidebar.markdown("---")

    # 1. Probe selector
    categories = ["QA", "Cloze", "Concept"]
    selected_category = st.sidebar.selectbox("Evaluation Probe", categories, index=0)
    norm_probe = normalize_probe_name(selected_category)

    # 2. Query available Run IDs for the selected probe
    available_runs = list_db_runs(norm_probe, db_path=trace_db_path)

    if not available_runs:
        st.sidebar.warning(f"No traces found in database for `{selected_category}`.")
        st.info(
            f"### No `{selected_category}` Traces in Database\n\n"
            f"Run evaluations or ingest trace logs via:\n"
            f"```bash\npython scripts/load_traces_to_db.py\n```\n\n"
            f"Database file: `{trace_db_path}`"
        )
        return

    run_label_map = {r["run_id"]: r["label"] for r in available_runs}
    run_ids = [r["run_id"] for r in available_runs]

    # Dropdown 1: Run ID
    selected_run_id = st.sidebar.selectbox(
        "Run ID",
        run_ids,
        format_func=lambda rid: run_label_map.get(rid, rid),
        index=0,
        help="Select evaluation run session.",
    )

    # Query checkpoints available for the selected Run ID
    available_checkpoints = list_db_checkpoints_for_run(
        probe=norm_probe,
        run_id=selected_run_id,
        db_path=trace_db_path,
    )

    if not available_checkpoints:
        st.sidebar.warning(f"No checkpoints found for Run ID `{selected_run_id}`.")
        return

    ckpt_map = {c["checkpoint"]: c for c in available_checkpoints}
    ckpt_numbers = [c["checkpoint"] for c in available_checkpoints]

    # Dropdown 2: Base Model / Reference (corresponding to Run ID)
    ref_options = [None] + ckpt_numbers
    default_base_idx = 0
    base_candidates = [c["checkpoint"] for c in available_checkpoints if c["is_base"]]
    if base_candidates:
        default_base_idx = ref_options.index(base_candidates[0])
    elif len(ckpt_numbers) > 1:
        default_base_idx = 1  # earliest checkpoint

    def format_base_option(c):
        if c is None:
            return "(None - Single Checkpoint View)"
        return ckpt_map[c]["label"]

    selected_base_ckpt = st.sidebar.selectbox(
        "Base Model / Reference",
        ref_options,
        format_func=format_base_option,
        index=default_base_idx,
        help="Select baseline reference checkpoint corresponding to selected Run ID.",
    )

    # Dropdown 3: Checkpoint (corresponding to Run ID)
    default_target_idx = len(ckpt_numbers) - 1  # newest checkpoint
    selected_target_ckpt = st.sidebar.selectbox(
        "Selected Checkpoint",
        ckpt_numbers,
        format_func=lambda c: ckpt_map[c]["label"],
        index=default_target_idx,
        help="Select checkpoint corresponding to selected Run ID to evaluate.",
    )

    target_info = {
        "run_id": selected_run_id,
        "checkpoint": selected_target_ckpt,
        "checkpoint_name": ckpt_map[selected_target_ckpt]["checkpoint_name"],
    }
    base_info = {
        "run_id": selected_run_id,
        "checkpoint": selected_base_ckpt,
        "checkpoint_name": ckpt_map[selected_base_ckpt]["checkpoint_name"] if selected_base_ckpt is not None else "None",
    } if selected_base_ckpt is not None else None

    st.sidebar.markdown("---")

    # Load DataFrames from SQLite
    target_df = load_probe_traces_from_db(
        probe=norm_probe,
        run_id=selected_run_id,
        checkpoint=selected_target_ckpt,
        db_path=trace_db_path,
    )

    base_df = load_probe_traces_from_db(
        probe=norm_probe,
        run_id=selected_run_id,
        checkpoint=selected_base_ckpt,
        db_path=trace_db_path,
    ) if selected_base_ckpt is not None else pd.DataFrame()

    diff_df = compute_trace_diff(base_df, target_df)

    if diff_df.empty:
        st.warning("Selected trace run contains no data rows.")
        return

    # 6. Sidebar Filters
    show_changed_only = st.sidebar.checkbox(
        "Show only traces changed from base",
        value=False,
        help="Filter items where the selected checkpoint generated a different answer than the baseline.",
    )

    # Subcategory filter
    available_subcats = sorted([c for c in diff_df["category_display"].unique() if str(c).strip()])
    if available_subcats:
        selected_subcat = st.sidebar.selectbox(
            "Filter Category / Cluster",
            ["All Categories"] + available_subcats,
        )
        if selected_subcat != "All Categories":
            diff_df = diff_df[diff_df["category_display"] == selected_subcat]

    # Error Category filter (dynamic from DB)
    available_error_cats = get_distinct_error_categories(norm_probe, db_path=trace_db_path)
    selected_error_filter = st.sidebar.selectbox(
        "Filter Error Category",
        ["All Error Categories"] + available_error_cats,
        index=0,
        help="Filter items by assigned error category in the database.",
    )
    if selected_error_filter != "All Error Categories":
        diff_df = diff_df[diff_df["error_category_target"].astype(str).str.strip() == selected_error_filter]

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

    # Header and Summary Metrics (Compact)
    st.markdown(
        f"<h3 style='margin: 0.3rem 0 0.5rem 0; font-size: 1.35rem; font-weight: 700;'>🔍 {selected_category} Probe Traces</h3>",
        unsafe_allow_html=True,
    )

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
    seq_num = int(row.get("seq_num", curr_idx + 1))
    category_val = row.get("category_display", "")
    prompt_text = row.get("prompt_display", "")
    target_answer_text = row.get("target_answer_display", "")

    # Top Prompt Card with explicit dark text color
    st.markdown(
        f"""
        <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; color: #1e293b; font-size: 0.95rem;">Sequence #{seq_num}</span>
                <span style="background: #e2e8f0; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; color: #334155; font-weight: 600;">Category: {category_val or 'Standard'}</span>
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a; margin-bottom: 4px;">Prompt:</div>
            <div style="background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 4px; padding: 10px; margin-bottom: 8px; font-family: monospace; font-size: 0.92rem; white-space: pre-wrap; word-break: break-word;">{prompt_text}</div>
            <div style="font-size: 0.9rem; font-weight: 600; color: #0f172a; margin-bottom: 4px;">Expected Target Answer:</div>
            <div style="background-color: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; border-radius: 4px; padding: 8px; font-family: monospace; font-size: 0.92rem; font-weight: 600; word-break: break-word;">{target_answer_text}</div>
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

    # Current Error Category in DB
    current_saved_cat = str(row.get("error_category_target", row.get("error_category", "")) or "").strip()
    target_res = str(row.get("result_target", "Fail")).strip().capitalize()
    if not current_saved_cat:
        current_saved_cat = "Pass" if target_res == "Pass" else "Fail"

    st.markdown(
        f"""
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 12px;">
            <span style="font-weight: 700; color: {status_color}; font-size: 1.0rem;">{status_label}</span>
            <span style="background: #e2e3e5; color: #383d41; padding: 2px 8px; border-radius: 4px; font-size: 0.82rem; font-weight: 600;">{output_diff_badge}</span>
            <span class="badge-category">Category: <strong>{current_saved_cat}</strong></span>
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
            <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px; background-color: #ffffff; min-height: 160px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="{b_badge_class}">{base_res}</span>
                    <span style="font-size: 0.85rem; color: #475569;">Score: <strong style="color: #0f172a;">{base_score:.4f}</strong></span>
                </div>
                <div style="font-family: monospace; white-space: pre-wrap; font-size: 0.92rem; color: #0f172a; word-break: break-word;">{base_out}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Right Column: Checkpoint Output
    with col_target:
        st.subheader("Checkpoint Output")
        tgt_out = str(row.get("model_output_target", "")).strip() or "(No output recorded)"
        tgt_score = row.get("matching_score_target", 0.0)
        tgt_res = str(row.get("result_target", "N/A")).strip().capitalize()
        
        t_badge_class = "badge-pass" if tgt_res == "Pass" else ("badge-fail" if tgt_res == "Fail" else "badge-neutral")
        
        # Border highlight if improved/regressed
        border_color = "#198754" if delta_status == "Improved" else ("#dc3545" if delta_status == "Regressed" else "#0d6efd")

        st.markdown(
            f"""
            <div style="border: 2px solid {border_color}; border-radius: 6px; padding: 12px; background-color: #ffffff; min-height: 160px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="{t_badge_class}">{tgt_res}</span>
                    <span style="font-size: 0.85rem; color: #475569;">Score: <strong style="color: #0f172a;">{tgt_score:.4f}</strong></span>
                </div>
                <div style="font-family: monospace; white-space: pre-wrap; font-size: 0.92rem; color: #0f172a; word-break: break-word;">{tgt_out}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Error Categorization & Annotation Card
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    with st.expander("🏷️ **Capture Error Category & Notes for this Item**", expanded=True):
        cat_options = list(available_error_cats)
        if current_saved_cat not in cat_options:
            cat_options.append(current_saved_cat)
        
        custom_opt = "+ Add New Custom Category..."
        dropdown_options = cat_options + [custom_opt]
        
        # Default index
        default_idx = cat_options.index(current_saved_cat) if current_saved_cat in cat_options else 0

        c_col1, c_col2, c_col3 = st.columns([3, 4, 2])

        with c_col1:
            selected_err_cat = st.selectbox(
                "Error Category",
                dropdown_options,
                index=default_idx,
                key=f"cat_select_{seq_num}_{target_info['run_id']}_{target_info['checkpoint']}",
            )
            
            final_category = selected_err_cat
            if selected_err_cat == custom_opt:
                custom_name = st.text_input(
                    "Enter Custom Category:",
                    key=f"custom_cat_input_{seq_num}_{target_info['run_id']}",
                    placeholder="e.g. Inconsistent Terminology",
                ).strip()
                if custom_name:
                    final_category = custom_name
                else:
                    final_category = "Fail"

        with c_col2:
            current_notes = str(row.get("notes_target", row.get("notes", "")) or "")
            notes_input = st.text_input(
                "Annotation Notes",
                value=current_notes,
                key=f"notes_input_{seq_num}_{target_info['run_id']}_{target_info['checkpoint']}",
                placeholder="Optional notes describing model behavior...",
            )

        with c_col3:
            st.write("")
            st.write("")
            if st.button("💾 Save Annotation", key=f"save_btn_{seq_num}", use_container_width=True):
                success = update_trace_annotation(
                    probe=norm_probe,
                    run_id=target_info["run_id"],
                    checkpoint=target_info["checkpoint"],
                    seq_num=seq_num,
                    error_category=final_category,
                    notes=notes_input,
                    db_path=trace_db_path,
                )
                if success:
                    st.toast(f"✅ Saved category '{final_category}' for Item #{seq_num}")
                    st.rerun()
                else:
                    st.error("Failed to update database record.")

    # Tabular Table View
    with st.expander("📋 View All Filtered Traces in Table"):
        display_cols = [
            "seq_num",
            "category_display",
            "prompt_display",
            "target_answer_display",
            "result_base",
            "result_target",
            "error_category_target",
            "delta_status",
            "is_output_changed",
            "model_output_base",
            "model_output_target",
        ]
        available_cols = [c for c in display_cols if c in diff_df.columns]
        st.dataframe(diff_df[available_cols], use_container_width=True)


if __name__ == "__main__":
    main()
