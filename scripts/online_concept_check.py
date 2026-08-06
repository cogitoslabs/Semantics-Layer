"""
online_concept_check.py — Interactive and CLI terminal tool for online concept probe check.

Loads model and tokenizer based on .env.common and PipelineConfig, takes prompt input
from terminal, and generates output using exact concept probe generation parameters.
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
from lib.s3_dapt.probes.concept_probe import generate_response
from lib.utils.logger import get_logger

logger = get_logger("scripts.online_concept_check")


def run_concept_check(prompt: str, model, tokenizer, cfg: PipelineConfig, device: torch.device) -> None:
    """Run model response generation using exact concept probe generation params and print output."""
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=cfg.probes.concept_max_new_tokens,
        device=device.type,
        max_length=cfg.probes.concept_max_seq_len,
    )
    print("\n" + "=" * 60)
    print(f"Input Prompt: {prompt}")
    print("-" * 60)
    print("Model Response:")
    print(response)
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Online Concept Probe Prompt Checker")
    parser.add_argument("--prompt", "-p", type=str, help="Concept prompt input (e.g. 'Explain synaptic plasticity')")
    args = parser.parse_args()

    cfg = PipelineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model '{cfg.model.base_model_name}' on device '{device}'...")

    model, tokenizer = load_model_and_tokenizer(cfg, device)
    model.eval()

    if args.prompt:
        run_concept_check(args.prompt, model, tokenizer, cfg, device)
    else:
        print("\n=== Online Concept Probe Interactive Session ===")
        print("Enter a concept prompt to generate a response.")
        print("Type 'exit' or 'quit' to end session.\n")
        while True:
            try:
                user_input = input("Concept Prompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting session.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting session.")
                break

            run_concept_check(user_input, model, tokenizer, cfg, device)


if __name__ == "__main__":
    main()
