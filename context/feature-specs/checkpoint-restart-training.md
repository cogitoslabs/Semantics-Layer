# Feature Spec: Checkpoint Restart & Resume Training

## Objective

Provide seamless restart/resume functionality for Domain Adaptive Pretraining (DAPT). If a long-running training job fails, crashes, or is interrupted, setting `RESTART_TRAINING_FROM_CHECKPOINT=True` enables resuming from the latest saved evaluation checkpoint, including model weights, optimizer/scheduler states, epoch/step positions, and historical evaluation probe metrics.

---

## Scope

This specification defines the additions and changes required across configuration, checkpoint management, and training execution:

1. **Environment & Configuration**:
   - Add `RESTART_TRAINING_FROM_CHECKPOINT` to `.env.common` and `.env.example`.
   - Update `ModelConfig` in `lib/utils/config.py` to parse `restart_from_checkpoint` (default: `False`).

2. **Checkpoint & State Management**:
   - Track `epoch` and `epoch_step` in `state` inside `dapt.py`.
   - Update `save_checkpoint` in `lib/utils/checkpoint.py` to include `scheduler` state dict alongside `optimizer`.
   - Add helper function `find_latest_checkpoint(checkpoint_dir)` in `lib/utils/checkpoint.py` to discover the most recent `.pt` file by timestamp or eval ID.
   - Update `load_checkpoint` in `lib/utils/checkpoint.py` to optionally accept and restore `scheduler` state dict.

3. **Pipeline Resuming Logic**:
   - In `run_dapt_pipeline_impl` (`lib/s3_dapt/dapt.py`), if `restart_from_checkpoint=True`:
     - Discover latest checkpoint via `find_latest_checkpoint(...)`.
     - If found, reload model, optimizer, scheduler, and `state`.
     - If not found, log a warning and fallback gracefully to training from scratch.

4. **DataLoader Skip Logic**:
   - In `run_training_loop` (`lib/s3_dapt/dapt.py`), start epoch iteration from `state.get("epoch", 0)`.
   - For the initial resumed epoch, skip batches already processed prior to the checkpoint (`batch_idx < start_step`).

5. **End-of-Training Best Checkpoint Export Consistency**:
   - In `handle_final_check` (`lib/s3_dapt/dapt.py`), call `model.save_pretrained(output_dir)` *after* `run_final_eval` reloads `best_ckpt` so the output directory always contains the best model.

---

## Technical Specifications

### 1. Environment Variable & Configuration
In `.env.common` and `.env.example`:
```env
RESTART_TRAINING_FROM_CHECKPOINT=False
```
In `lib/utils/config.py` (`ModelConfig`):
```python
restart_from_checkpoint: bool = field(default_factory=lambda: get("RESTART_TRAINING_FROM_CHECKPOINT", False, bool))
```

### 2. State & Checkpoint Helper Updates
In `lib/utils/checkpoint.py`:
- `find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]`:
  Scans `checkpoint_dir` for `dapt_eval_*.pt` files and returns the path with the highest modification time or eval ID index.
- Update `save_checkpoint` and `load_checkpoint` signatures and payloads to include optional `scheduler`.

### 3. Training Loop Resume Behavior
In `lib/s3_dapt/dapt.py`:
- Initialize state with `epoch: 0` and `epoch_step: 0`.
- Maintain `state["epoch"]` and `state["epoch_step"]` during training loop execution.
- Resume training loop at `start_epoch` and skip batches `0..start_step-1` on the first resumed epoch.
