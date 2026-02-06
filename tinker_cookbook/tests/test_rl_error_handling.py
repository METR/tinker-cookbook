"""Tests for RL worker error handling when API calls fail."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tinker

from tinker_cookbook.rl.train import do_group_rollout_and_filter_constant_reward


@pytest.fixture
def mock_sampling_client():
    return MagicMock(spec=tinker.SamplingClient)


@pytest.fixture
def mock_env_group_builder():
    builder = MagicMock()
    builder.logging_tags.return_value = ["test_env"]
    return builder


@pytest.mark.asyncio
async def test_api_status_error_returns_none(mock_sampling_client, mock_env_group_builder):
    """When do_group_rollout raises APIStatusError, should return None instead of crashing."""
    with patch(
        "tinker_cookbook.rl.train.do_group_rollout",
        new_callable=AsyncMock,
        side_effect=tinker.BadRequestError(
            message="context length exceeded",
            response=MagicMock(status_code=400),
            body=None,
        ),
    ):
        result = await do_group_rollout_and_filter_constant_reward(
            sampling_client=mock_sampling_client,
            env_group_builder=mock_env_group_builder,
            max_tokens=1024,
            temperature=1.0,
            do_remove_constant_reward_groups=False,
        )

    assert result is None


@pytest.mark.asyncio
async def test_non_api_error_propagates(mock_sampling_client, mock_env_group_builder):
    """Non-API exceptions (bugs in env code) should still crash loudly."""
    with patch(
        "tinker_cookbook.rl.train.do_group_rollout",
        new_callable=AsyncMock,
        side_effect=KeyError("unexpected bug"),
    ):
        with pytest.raises(KeyError, match="unexpected bug"):
            await do_group_rollout_and_filter_constant_reward(
                sampling_client=mock_sampling_client,
                env_group_builder=mock_env_group_builder,
                max_tokens=1024,
                temperature=1.0,
                do_remove_constant_reward_groups=False,
            )
