"""
online_cloze_check.py — Interactive and CLI terminal tool for online cloze probe check.

Loads model and tokenizer based on .env and PipelineConfig, takes prompt input
from terminal, and generates completions using exact cloze probe generation parameters.
"""

import argparse
import sys
from pathlib import Path

# Ensure workspace root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from lib.utils.config import PipelineConfig
from lib.s3_dapt.model_utils import load_model_and_tokenizer
from lib.s3_dapt.probes.cloze_probe import generate_topk_completions, format_cloze_prompt
from lib.utils.logger import get_logger

logger = get_logger("scripts.online_cloze_check")


def run_cloze_check(prompt: str, model, tokenizer, cfg: PipelineConfig, device: torch.device) -> None:
    """Format prompt, run top-k cloze completion using probe params, and print results."""
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
    print("\n" + "=" * 60)
    print(f"Input Prompt : {prompt}")
    print("-" * 60)
    print(f"Formatted Prompt:\n{formatted_prompt}")
    print("-" * 60)
    print(f"Model Top-{cfg.probes.cloze_top_k} Completions:")
    for idx, comp in enumerate(completions, 1):
        print(f"  [{idx}] {comp}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Online Cloze Probe Prompt Checker")
    parser.add_argument("--prompt", "-p", type=str, help="Cloze prompt input (e.g. 'Dopamine is a ___')")
    args = parser.parse_args()

    cfg = PipelineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model '{cfg.model.base_model_name}' on device '{device}'...")

    model, tokenizer = load_model_and_tokenizer(cfg, device)
    model.eval()

    if args.prompt:
        run_cloze_check(args.prompt, model, tokenizer, cfg, device)
    else:
        print("\n=== Online Cloze Probe Interactive Session ===")
        print("Enter a cloze prompt (e.g. 'A neurotransmitter involved in reward is ___').")
        print("Type 'exit' or 'quit' to end session.\n")
        while True:
            try:
                user_input = input("Cloze Prompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting session.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting session.")
                break

            run_cloze_check(user_input, model, tokenizer, cfg, device)


if __name__ == "__main__":
    main()
