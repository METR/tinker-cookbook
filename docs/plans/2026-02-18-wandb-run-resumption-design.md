# Wandb Run Resumption on Checkpoint Resume

## Problem

When training is resumed from a checkpoint, `setup_logging` always creates a new wandb run via `wandb.init()`. The original and resumed runs are completely disconnected, making it hard to track a single training run's full history.

## Solution

Persist the wandb run ID to a file (`wandb_run_id.txt`) in the log directory. On resume, read it back and pass it to `wandb.init(id=..., resume="allow")` to continue the same run.

## Changes

Only `tinker_cookbook/utils/ml_log.py`:

- `WandbLogger.__init__` accepts an optional `log_dir` parameter.
- On init: if `log_dir` is provided, check for `wandb_run_id.txt`. If found, pass the saved ID with `resume="allow"`.
- After successful init: write `self.run.id` to `wandb_run_id.txt`.

No changes to training scripts, `checkpoint_utils`, or any other files.

## Behavior Matrix

| Scenario | Behavior |
|---|---|
| First run, no wandb | No file written |
| First run, with wandb | New run created, ID saved |
| Resume, file exists | `resume="allow"` continues the run |
| Resume, file missing | Warning logged, new run created, new ID saved |
| Resume, run deleted from wandb UI | `resume="allow"` creates a new run with that ID |
