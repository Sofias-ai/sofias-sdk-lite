"""``usage_state_from_list`` collapses a turn's usage list into ``state.usage``."""

from __future__ import annotations

from sofias_sdk_lite import usage_state_from_list


def test_none_returns_none() -> None:
    assert usage_state_from_list(None) is None


def test_empty_list_returns_none() -> None:
    assert usage_state_from_list([]) is None


def test_single_entry_is_passed_through() -> None:
    result = usage_state_from_list(
        [{"prompt_tokens": 100, "completion_tokens": 20, "model_requested": "gpt-4o"}]
    )
    assert result == {"prompt_tokens": 100, "completion_tokens": 20, "model": "gpt-4o"}


def test_multiple_entries_sum_tokens_and_pick_the_generation_model() -> None:
    result = usage_state_from_list(
        [
            {"prompt_tokens": 40, "completion_tokens": 0, "model_requested": "embedding-model"},
            {"prompt_tokens": 120, "completion_tokens": 30, "model_requested": "chat-model"},
        ]
    )
    assert result == {"prompt_tokens": 160, "completion_tokens": 30, "model": "chat-model"}


def test_object_style_entries_are_supported() -> None:
    class Entry:
        prompt_tokens = 50
        completion_tokens = 10
        model_requested = "claude"

    assert usage_state_from_list([Entry()]) == {
        "prompt_tokens": 50,
        "completion_tokens": 10,
        "model": "claude",
    }


def test_missing_fields_default_to_zero_and_empty_model() -> None:
    assert usage_state_from_list([{}]) == {"prompt_tokens": 0, "completion_tokens": 0, "model": ""}


def test_garbage_token_values_count_as_zero() -> None:
    assert usage_state_from_list([{"prompt_tokens": "n/a", "completion_tokens": None}]) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "model": "",
    }
