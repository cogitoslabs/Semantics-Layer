"""
online_probes.py — Streamlit Web UI for Online Probe Checks (Cloze Probe, QA Probe & Concept Probe)
Supports side-by-side comparison between Base Model and saved DAPT Checkpoints.
"""

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import streamlit as st

from lib.utils.config import PipelineConfig
from lib.utils.checkpoint import load_checkpoint
from lib.s3_dapt.model_utils import load_model_and_tokenizer
from lib.s3_dapt.probes.cloze_probe import generate_topk_completions, format_cloze_prompt
from lib.s3_dapt.probes.concept_probe import generate_response
from lib.s3_dapt.probes.qa_probe import score_choices_by_logprob


def list_available_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """Scan checkpoint directory and return list of .pt checkpoints sorted newest first."""
    if not checkpoint_dir.exists():
        return []
    ckpts = sorted(checkpoint_dir.glob("dapt_eval_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts


@st.cache_resource
def load_cached_model(model_source: str, checkpoint_filename: Optional[str] = None) -> Tuple[Any, Any, PipelineConfig, torch.device]:
    """
    Load model and tokenizer into memory, cached by model source and checkpoint filename.
    """
    cfg = PipelineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(cfg, device)

    if model_source == "Checkpoint" and checkpoint_filename:
        ckpt_path = cfg.model.checkpoint_dir / checkpoint_filename
        if ckpt_path.exists():
            load_checkpoint(ckpt_path, model)
        else:
            raise FileNotFoundError(f"Selected checkpoint file not found: {ckpt_path}")

    model.eval()
    return model, tokenizer, cfg, device


def run_probe_logic(
    probe_type: str,
    prompt: str,
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: PipelineConfig,
    device: torch.device,
    choices: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run model generation / evaluation logic depending on selected probe type."""
    if probe_type == "Cloze Probe":
        formatted_prompt = format_cloze_prompt(prompt)
        completions = generate_topk_completions(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            k=cfg.probes.cloze_top_k,
            max_new_tokens=cfg.probes.cloze_max_new_tokens,
            device=device.type,
            max_length=cfg.probes.cloze_max_seq_len,
        )
        return {
            "type": "cloze",
            "formatted_prompt": formatted_prompt,
            "completions": completions,
            "top_k": cfg.probes.cloze_top_k,
        }
    elif probe_type == "QA Probe":
        valid_choices = [c.strip() for c in (choices or []) if c.strip()]
        if not valid_choices:
            raise ValueError("No valid choices provided for QA Probe.")
        
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        predicted_idx = score_choices_by_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            choices=valid_choices,
            device=device.type,
            max_length=cfg.probes.qa_max_seq_len,
        )
        return {
            "type": "qa",
            "question": prompt,
            "formatted_prompt": formatted_prompt,
            "choices": valid_choices,
            "predicted_idx": int(predicted_idx),
            "predicted_choice": valid_choices[predicted_idx],
        }
    else:
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=cfg.probes.concept_max_new_tokens,
            device=device.type,
            max_length=cfg.probes.concept_max_seq_len,
        )
        return {
            "type": "concept",
            "response": response,
            "max_new_tokens": cfg.probes.concept_max_new_tokens,
        }


def render_probe_output(result: Dict[str, Any]) -> None:
    """Render probe results formatted for UI display."""
    if result["type"] == "cloze":
        st.markdown(f"**Top-{result['top_k']} Cloze Completions:**")
        for idx, comp in enumerate(result["completions"], 1):
            st.info(f"**[{idx}]** {comp}")

        with st.expander("View Formatted Few-Shot Prompt"):
            st.code(result["formatted_prompt"], language="text")

    elif result["type"] == "qa":
        pred_letter = chr(65 + result["predicted_idx"])
        st.success(f"**Predicted Answer (Choice {pred_letter}):** {result['predicted_choice']}")
        
        st.markdown("**Candidate Choice Rankings:**")
        for idx, choice in enumerate(result["choices"]):
            letter = chr(65 + idx)
            if idx == result["predicted_idx"]:
                st.markdown(f"🔹 **Choice {letter} (Selected):** `{choice}`")
            else:
                st.markdown(f"▫️ **Choice {letter}:** {choice}")

        with st.expander("View Formatted QA Prompt"):
            st.code(result["formatted_prompt"], language="text")

    else:
        st.markdown("**Generated Response:**")
        st.success(result["response"])


def main():
    st.set_page_config(
        page_title="Online Probe Checker",
        page_icon="🧠",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem !important;
                padding-bottom: 1rem !important;
            }
            h3 {
                margin-top: 0.2rem !important;
                margin-bottom: 0.5rem !important;
                font-size: 1.35rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cfg_tmp = PipelineConfig()
    checkpoints = list_available_checkpoints(cfg_tmp.model.checkpoint_dir)
    ckpt_options = [ckpt.name for ckpt in checkpoints]

    # Sidebar: Probe and Checkpoint selection
    st.sidebar.header("Configuration")
    probe_type = st.sidebar.selectbox(
        "Probe",
        ["Cloze Probe", "QA Probe", "Concept Probe"],
        help="Select Cloze Probe (fill-in-the-blank), QA Probe (multiple choice scoring), or Concept Probe (freeform response)",
    )

    selected_ckpt_filename = None
    if ckpt_options:
        selected_ckpt_filename = st.sidebar.selectbox(
            "Checkpoint",
            ckpt_options,
            help="Select a checkpoint to compare against the Base Model.",
        )
    else:
        st.sidebar.selectbox(
            "Checkpoint",
            ["(No checkpoints found)"],
            disabled=True,
            help=f"No checkpoints found in `{cfg_tmp.model.checkpoint_dir}`",
        )

    st.sidebar.markdown("---")
    st.sidebar.header("Model Information")
    st.sidebar.write(f"**Base Model**: `{cfg_tmp.model.base_model_name}`")
    if selected_ckpt_filename:
        st.sidebar.write(f"**Selected Checkpoint**: `{selected_ckpt_filename}`")
    else:
        st.sidebar.write("**Selected Checkpoint**: `None`")

    # Main Area: Prompt & Action
    if probe_type == "Cloze Probe":
        default_prompt = "A neurotransmitter involved in reward and motivation is ___."
    elif probe_type == "QA Probe":
        default_prompt = "What brain structure is primarily responsible for forming new long-term declarative memories?"
    else:
        default_prompt = "Explain the functional role of the hippocampus in memory consolidation."

    col_prompt, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    with col_prompt:
        user_prompt = st.text_area(
            "Question / Prompt",
            value=default_prompt,
            height=100,
            placeholder="Enter your prompt or question here...",
        )
    with col_btn:
        generate_clicked = st.button("Generate Response", type="primary", use_container_width=True)

    qa_choices = []
    if probe_type == "QA Probe":
        st.markdown("**Answer Choices:**")
        col1, col2 = st.columns(2)
        with col1:
            choice_a = st.text_input("Choice A", value="Hippocampus")
            choice_b = st.text_input("Choice B", value="Cerebellum")
        with col2:
            choice_c = st.text_input("Choice C", value="Occipital lobe")
            choice_d = st.text_input("Choice D", value="Medulla oblongata")
        qa_choices = [choice_a, choice_b, choice_c, choice_d]

    if generate_clicked:
        if not user_prompt.strip():
            st.warning("Please enter a valid prompt.")
            return

        # 1. Base Model Inference
        with st.spinner("Generating output from Base Model..."):
            base_model, base_tok, cfg, device = load_cached_model("Base Model", None)
            base_result = run_probe_logic(
                probe_type=probe_type,
                prompt=user_prompt.strip(),
                model=base_model,
                tokenizer=base_tok,
                cfg=cfg,
                device=device,
                choices=qa_choices if probe_type == "QA Probe" else None,
            )

        # 2. Checkpoint Model Inference (if available)
        ckpt_result = None
        if selected_ckpt_filename:
            with st.spinner(f"Generating output from Checkpoint ({selected_ckpt_filename})..."):
                ckpt_model, ckpt_tok, cfg, device = load_cached_model("Checkpoint", selected_ckpt_filename)
                ckpt_result = run_probe_logic(
                    probe_type=probe_type,
                    prompt=user_prompt.strip(),
                    model=ckpt_model,
                    tokenizer=ckpt_tok,
                    cfg=cfg,
                    device=device,
                    choices=qa_choices if probe_type == "QA Probe" else None,
                )

        # 3. Render side-by-side output columns without redundant top title
        st.markdown("---")
        col_base_out, col_ckpt_out = st.columns(2)

        with col_base_out:
            st.subheader("Base Model")
            render_probe_output(base_result)

        with col_ckpt_out:
            if selected_ckpt_filename and ckpt_result:
                st.subheader(f"Checkpoint ({selected_ckpt_filename})")
                render_probe_output(ckpt_result)
            else:
                st.subheader("Selected Checkpoint")
                st.info("No checkpoint available or selected.")


if __name__ == "__main__":
    main()
