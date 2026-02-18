# Wandb Run Resumption Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When resuming training from a checkpoint, continue the existing wandb run instead of creating a new one.

**Architecture:** Persist the wandb run ID to `wandb_run_id.txt` in the log directory on first init. On subsequent inits in the same log directory, read the file and pass `id=..., resume="must"` to `wandb.init()`. All changes are contained in `tinker_cookbook/utils/ml_log.py`.

**Tech Stack:** wandb, pytest, pytest-mock

**Design doc:** `docs/plans/2026-02-18-wandb-run-resumption-design.md`

---

### Task 1: Add tests for WandbLogger run ID persistence

**Files:**
- Create: `tinker_cookbook/tests/test_ml_log.py`

**Step 1: Write failing tests**

```python
"""Tests for wandb run resumption in ml_log."""

import os
from unittest.mock import MagicMock, patch

import pytest


WANDB_RUN_ID_FILE = "wandb_run_id.txt"


@pytest.fixture
def mock_wandb():
    """Mock wandb module and its init function."""
    mock_run = MagicMock()
    mock_run.id = "test-run-id-abc123"
    mock_run.url = "https://wandb.ai/test/run"

    with patch.dict(os.environ, {"WANDB_API_KEY": "fake-key"}):
        with patch("tinker_cookbook.utils.ml_log._wandb_available", True):
            with patch("tinker_cookbook.utils.ml_log.wandb") as mock_wandb_module:
                mock_wandb_module.init.return_value = mock_run
                yield mock_wandb_module, mock_run


class TestWandbRunIdPersistence:
    def test_first_run_saves_run_id_to_file(self, tmp_path, mock_wandb):
        """First run should save the wandb run ID to wandb_run_id.txt."""
        from tinker_cookbook.utils.ml_log import WandbLogger

        mock_wandb_module, mock_run = mock_wandb

        WandbLogger(project="test", log_dir=tmp_path)

        run_id_file = tmp_path / WANDB_RUN_ID_FILE
        assert run_id_file.exists()
        assert run_id_file.read_text().strip() == "test-run-id-abc123"

        # Should NOT have passed resume args on first run
        mock_wandb_module.init.assert_called_once()
        call_kwargs = mock_wandb_module.init.call_args[1]
        assert "id" not in call_kwargs or call_kwargs["id"] is None
        assert "resume" not in call_kwargs or call_kwargs["resume"] is None

    def test_resume_uses_saved_run_id(self, tmp_path, mock_wandb):
        """When wandb_run_id.txt exists, should resume with that ID."""
        from tinker_cookbook.utils.ml_log import WandbLogger

        mock_wandb_module, mock_run = mock_wandb

        # Simulate a previous run having saved its ID
        run_id_file = tmp_path / WANDB_RUN_ID_FILE
        run_id_file.write_text("previous-run-id-xyz789")

        WandbLogger(project="test", log_dir=tmp_path)

        mock_wandb_module.init.assert_called_once()
        call_kwargs = mock_wandb_module.init.call_args[1]
        assert call_kwargs["id"] == "previous-run-id-xyz789"
        assert call_kwargs["resume"] == "must"

    def test_no_log_dir_skips_persistence(self, mock_wandb):
        """When log_dir is None, should not try to read/write run ID file."""
        from tinker_cookbook.utils.ml_log import WandbLogger

        mock_wandb_module, mock_run = mock_wandb

        WandbLogger(project="test", log_dir=None)

        mock_wandb_module.init.assert_called_once()
        call_kwargs = mock_wandb_module.init.call_args[1]
        assert "id" not in call_kwargs or call_kwargs["id"] is None
        assert "resume" not in call_kwargs or call_kwargs["resume"] is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tinker_cookbook/tests/test_ml_log.py -v`
Expected: FAIL — `test_first_run_saves_run_id_to_file` fails because no file is written; `test_resume_uses_saved_run_id` fails because `id` and `resume` are not passed.

---

### Task 2: Implement run ID persistence in WandbLogger

**Files:**
- Modify: `tinker_cookbook/utils/ml_log.py:201-227` (WandbLogger.__init__)

**Step 1: Implement the change**

Replace `WandbLogger.__init__` with:

```python
WANDB_RUN_ID_FILE = "wandb_run_id.txt"


class WandbLogger(Logger):
    """Logger for Weights & Biases."""

    def __init__(
        self,
        project: str | None = None,
        config: Any | None = None,
        log_dir: str | Path | None = None,
        wandb_name: str | None = None,
    ):
        if not _wandb_available:
            raise ImportError(
                "wandb is not installed. Please install it with: "
                "pip install wandb (or uv add wandb)"
            )

        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("WANDB_API_KEY environment variable not set")

        assert wandb is not None  # For type checker

        # Check for saved run ID to resume
        resume_run_id: str | None = None
        log_dir_path = Path(log_dir) if log_dir else None
        if log_dir_path is not None:
            run_id_file = log_dir_path / WANDB_RUN_ID_FILE
            if run_id_file.exists():
                resume_run_id = run_id_file.read_text().strip()
                logger.info(f"Found existing wandb run ID: {resume_run_id}, will resume")

        self.run = wandb.init(
            project=project,
            config=dump_config(config) if config else None,
            dir=str(log_dir) if log_dir else None,
            name=wandb_name,
            id=resume_run_id,
            resume="must" if resume_run_id else None,
        )

        # Persist run ID for future resumption
        if log_dir_path is not None and self.run is not None:
            run_id_file = log_dir_path / WANDB_RUN_ID_FILE
            run_id_file.write_text(self.run.id)
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tinker_cookbook/tests/test_ml_log.py -v`
Expected: All 3 tests PASS.

**Step 3: Run linting and type checking**

Run: `uv run ruff check tinker_cookbook/utils/ml_log.py && uv run ruff format tinker_cookbook/utils/ml_log.py`

**Step 4: Commit**

```bash
git add tinker_cookbook/utils/ml_log.py tinker_cookbook/tests/test_ml_log.py
git commit -m "feat(ml_log): resume wandb run on checkpoint resume"
```

---

### Task 3: Verify no regressions in existing tests

**Files:** None (verification only)

**Step 1: Run existing tests**

Run: `uv run pytest tinker_cookbook/tests/test_renderers.py tinker_cookbook/tests/test_utils.py -v`
Expected: All PASS.

**Step 2: Run full linting**

Run: `uv run ruff check . && uv run basedpyright tinker_cookbook/utils/ml_log.py tinker_cookbook/tests/test_ml_log.py`
Expected: Clean.
