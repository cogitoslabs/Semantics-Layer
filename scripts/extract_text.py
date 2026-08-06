"""
extract_text.py — Command Line Interface script to scan a directory, extract main narrative
text from PDF files using Docling, and save clean .txt outputs.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path to enable absolute imports when run as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.utils.pdf_utils import extract_main_text_from_pdfs
from lib.utils.logger import setup_logger
from lib.utils.config import LoggingConfig


def main():
    parser = argparse.ArgumentParser(description="Extract main narrative text from PDF files in a directory.")
    parser.add_argument("input_dir", type=str, help="Directory containing PDF files.")
    parser.add_argument("output_path", type=str, nargs="?", default=None, help="Optional output .txt file path or output directory. Defaults to input_dir.")
    
    args = parser.parse_args()
    
    output_path = args.output_path if args.output_path else args.input_dir
    
    # Configure console output logging
    setup_logger("pdf_utils", LoggingConfig())
    
    try:
        extract_main_text_from_pdfs(
            input_dir=args.input_dir,
            output_path=output_path,
        )
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
