# Feature Spec: PEFT-DAPT (Parameter-Efficient Continued Pretraining)

## Objective

Introduce an option to run Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) adapters during Domain Adaptive Pretraining (DAPT). This avoids training the full model parameters, reducing GPU memory footprint and enabling efficient continued pretraining on specialized corpora.

---

## Scope

This specification defines the additions and changes required in the configuration system and DAPT training/evaluation code to support PEFT-DAPT:
1. **Toggle in Environment**: Introduce `PEFT_DAPT` and associated LoRA hyperparameters in `.env.common` (and `.env.example`).
2. **Configuration Class Update**: Parse and validate the new LoRA parameters in `lib/utils/config.py`.
3. **Model Loading & Wrapping**: Integrate PEFT (`get_peft_model`) in `lib/s3_dapt/model_utils.py` to wrap the base causal LM with adapters when `PEFT_DAPT` is enabled.
4. **Optimizer Optimization**: Adjust `init_optimizer_scheduler` in `lib/s3_dapt/training_helpers.py` to only pass trainable parameters to the optimizer (yielding significant VRAM savings).
5. **Checkpoint Loading**: Update the checkpoint loader in `lib/utils/checkpoint.py` to support non-strict loading when a `PeftModel` is used (since base weights are frozen and not stored in checkpoints).
6. **Inference & Log Failures Loader**: Modify `run_inference_and_log_failures` in `lib/s3_dapt/evaluation/eval_runner.py` to correctly load base model + adapter weights when `PEFT_DAPT` is enabled.

---

## Technical Specifications

### 1. Environment & Configuration Settings
Add the following configuration options under `ModelConfig` in `lib/utils/config.py`:
- `PEFT_DAPT` (`peft_dapt`): `bool` (default: `False`). Toggles PEFT adapter wrapping.
- `LORA_R` (`lora_r`): `int` (default: `16`). LoRA rank.
- `LORA_ALPHA` (`lora_alpha`): `int` (default: `32`). LoRA scaling factor.
- `LORA_DROPOUT` (`lora_dropout`): `float` (default: `0.05`). Dropout probability for LoRA layers.
- `LORA_TARGET_MODULES` (`lora_target_modules`): `List[str]` (default: `["q_proj", "v_proj", "k_proj", "o_proj"]`). Modules to apply LoRA to.

### 2. Model Wrapping (LoRA)
In `lib/s3_dapt/model_utils.py`, when `cfg.model.peft_dapt` is `True`:
1. Check that `peft` is installed.
2. Initialize `LoraConfig` using the parsed parameters.
3. Wrap the base model with `get_peft_model(model, peft_config)`.
4. Log the count and ratio of trainable parameters.

### 3. Training Optimization (Optimizer & Memory)
In `lib/s3_dapt/training_helpers.py`:
- Filter parameters passed to the optimizer using `[p for p in model.parameters() if p.requires_grad]`. This avoids allocating optimizer state tracking for frozen parameters, preventing VRAM bloat.

### 4. Checkpoint Management
In `lib/utils/checkpoint.py`:
- When loading a checkpoint, check if `model` is an instance of `peft.PeftModel`.
- If it is, call `model.load_state_dict(payload["model_state_dict"], strict=False)`.

### 5. Final Failure Logging
In `lib/s3_dapt/evaluation/eval_runner.py`:
- When running `run_inference_and_log_failures`, check if `cfg.model.peft_dapt` is `True`.
- If so, load the base model from `cfg.model.base_model_name` first, and then load the adapter weights using `PeftModel.from_pretrained(base_model, model_dir)`.
