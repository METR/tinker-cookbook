"""Tests for wandb run resumption in ml_log."""

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from tinker_cookbook.utils.ml_log import WANDB_RUN_ID_FILE, WandbLogger


@pytest.fixture
def mock_wandb(mocker: MockerFixture):
    mock_run = MagicMock()
    mock_run.id = "test-run-id-abc123"
    mock_run.url = "https://wandb.ai/test/run"

    mocker.patch.dict("os.environ", {"WANDB_API_KEY": "fake-key"})
    mocker.patch("tinker_cookbook.utils.ml_log._wandb_available", True)
    mock_wandb_module = mocker.patch("tinker_cookbook.utils.ml_log.wandb")
    mock_wandb_module.init.return_value = mock_run
    yield mock_wandb_module, mock_run


class TestWandbRunIdPersistence:
    def test_first_run_saves_run_id_to_file(self, tmp_path, mock_wandb):
        """First run should save the wandb run ID to wandb_run_id.txt."""
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
        mock_wandb_module, mock_run = mock_wandb

        # Simulate a previous run having saved its ID
        run_id_file = tmp_path / WANDB_RUN_ID_FILE
        run_id_file.write_text("previous-run-id-xyz789")

        WandbLogger(project="test", log_dir=tmp_path)

        mock_wandb_module.init.assert_called_once()
        call_kwargs = mock_wandb_module.init.call_args[1]
        assert call_kwargs["id"] == "previous-run-id-xyz789"
        assert call_kwargs["resume"] == "allow"

    def test_no_log_dir_skips_persistence(self, mock_wandb):
        """When log_dir is None, should not try to read/write run ID file."""
        mock_wandb_module, mock_run = mock_wandb

        WandbLogger(project="test", log_dir=None)

        mock_wandb_module.init.assert_called_once()
        call_kwargs = mock_wandb_module.init.call_args[1]
        assert "id" not in call_kwargs or call_kwargs["id"] is None
        assert "resume" not in call_kwargs or call_kwargs["resume"] is None
