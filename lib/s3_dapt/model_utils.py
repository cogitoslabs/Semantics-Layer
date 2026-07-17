import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.utils import DAPTConfig

def load_model_and_tokenizer(cfg: DAPTConfig, device: torch.device):
    """Load model and tokenizer with GPU optimizations if applicable."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        attn_implementation = "sdpa"
    else:
        dtype = torch.float32
        attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_name,
        dtype=dtype,
        attn_implementation=attn_implementation
    )
    model.to(device)

    # Enable gradient checkpointing if configured
    if getattr(cfg.model, "gradient_checkpointing", False):
        if getattr(cfg.model, "peft_dapt", False):
            # PEFT requires enable_input_require_grads() to force gradients on input embeddings
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    if getattr(cfg.model, "peft_dapt", False):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "peft library is not installed, but PEFT_DAPT=True is requested. "
                "Please run `uv add peft`."
            )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            bias="none",
            target_modules=cfg.model.lora_target_modules,
        )
        model = get_peft_model(model, peft_config)
        
        try:
            from lib.utils.logger import get_logger
            logger = get_logger(__name__)
            trainable_params, all_params = model.get_nb_trainable_parameters()
            logger.info(
                f"PEFT-DAPT enabled. Trainable params: {trainable_params:,} / {all_params:,} "
                f"({100 * trainable_params / all_params:.4f}%)"
            )
        except Exception:
            pass
        model.to(device)

    return model, tokenizer
