import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.utils import DAPTConfig

def load_model_and_tokenizer(cfg: DAPTConfig, device: torch.device):
    """Load model and tokenizer with GPU optimizations if applicable."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        attn_implementation = "sdpa"
    else:
        torch_dtype = torch.float32
        attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_name,
        dtype=torch_dtype,
        attn_implementation=attn_implementation
    )
    model.to(device)
    return model, tokenizer
