"""Tests for contracts: strict validation, node contracts, agent envelopes."""

from __future__ import annotations

import pytest

from sofias_sdk_lite.contracts import (
    AgentMessage,
    AgentResponse,
    InputContract,
    NodeContract,
    OutputContract,
    ResponseStatus,
    StrictContract,
)
from sofias_sdk_lite.errors import InputValidationError, OutputValidationError


class DummyInput(InputContract):
    value: int


class DummyOutput(OutputContract):
    doubled: int


def test_strict_contract_silently_drops_unknown_fields():
    """StrictContract uses extra="ignore", not "forbid": unknown fields are
    dropped rather than rejected. This is intentional upstream behavior
    (matches the internal SDK) — see strict_contract.py's docstring."""

    class Strict(StrictContract):
        name: str

    instance = Strict(name="ok", extra_field="dropped")
    assert instance.name == "ok"
    assert not hasattr(instance, "extra_field")


def test_strict_contract_still_validates_declared_field_types():
    class Strict(StrictContract):
        name: str

    with pytest.raises(Exception):
        Strict(name=123)  # strict=True disables int -> str coercion


def test_node_contract_validate_input_accepts_valid_data():
    contract = NodeContract(input_schema=DummyInput, output_schema=DummyOutput)
    result = contract.validate_input({"value": 5})
    assert result.value == 5


def test_node_contract_validate_input_raises_on_invalid_data():
    contract = NodeContract(input_schema=DummyInput, output_schema=DummyOutput)
    with pytest.raises(InputValidationError):
        contract.validate_input({"value": "not an int"})


def test_node_contract_validate_output_raises_on_invalid_data():
    contract = NodeContract(input_schema=DummyInput, output_schema=DummyOutput)
    with pytest.raises(OutputValidationError):
        contract.validate_output({"doubled": "nope"})


def test_agent_message_wraps_typed_content():
    message = AgentMessage[DummyInput](content=DummyInput(value=3), conversation_id="conv-1")
    assert message.content.value == 3
    assert message.conversation_id == "conv-1"


def test_agent_response_defaults_to_success():
    response = AgentResponse(content={"doubled": 10}, agent_name="test", agent_version="0.1.0")
    assert response.status == ResponseStatus.SUCCESS
    assert response.content == {"doubled": 10}
