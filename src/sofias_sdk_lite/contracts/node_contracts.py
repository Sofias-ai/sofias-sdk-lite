"""Contract definitions for node and agent input/output validation.

This module provides the base classes and contracts used to define
and validate the input/output schemas of nodes and agents.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from sofias_sdk_lite.contracts.strict_contract import StrictContract
from sofias_sdk_lite.errors.exceptions import InputValidationError, OutputValidationError

__all__ = [
    "InputContract",
    "OutputContract",
    "InputT",
    "OutputT",
    "NodeContract",
    "AgentContract",
]


class InputContract(StrictContract):
    """Base class for node input contracts.

    Agents should inherit from this class to define their input schemas.
    Inherits strict validation from StrictContract.
    """

    pass


class OutputContract(StrictContract):
    """Base class for node output contracts.

    Agents should inherit from this class to define their output schemas.
    Inherits strict validation from StrictContract.
    """

    pass


InputT = TypeVar("InputT", bound=InputContract)
OutputT = TypeVar("OutputT", bound=OutputContract)


def _validate_with_model(
    model: type[BaseModel],
    data: dict[str, Any],
    *,
    is_input: bool = True,
) -> Any:
    """Validate data against a Pydantic model.

    Args:
        model: The Pydantic model class to validate against.
        data: The data dictionary to validate.
        is_input: If True, raises InputValidationError; otherwise OutputValidationError.

    Returns:
        The validated model instance.

    Raises:
        InputValidationError: If validation fails and is_input is True.
        OutputValidationError: If validation fails and is_input is False.
    """
    try:
        return model.model_validate(data)
    except ValidationError as e:
        error_cls = InputValidationError if is_input else OutputValidationError
        raise error_cls(
            schema_name=model.__name__,
            data=data,
            validation_errors=e.errors(),
        ) from e


class NodeContract(Generic[InputT, OutputT]):
    """Encapsulates the input/output contract pair for a node.

    Provides validation methods to verify data against the defined schemas.
    """

    def __init__(
        self,
        input_schema: type[InputT],
        output_schema: type[OutputT],
    ) -> None:
        """Initialize the node contract.

        Args:
            input_schema: The Pydantic model class for input validation.
            output_schema: The Pydantic model class for output validation.
        """
        self.input_schema = input_schema
        self.output_schema = output_schema

    def validate_input(self, data: dict[str, Any]) -> InputT:
        """Validate input data against the input schema.

        Args:
            data: The data dictionary to validate.

        Returns:
            The validated input model instance.

        Raises:
            InputValidationError: If validation fails.
        """
        return _validate_with_model(self.input_schema, data, is_input=True)

    def validate_output(self, data: dict[str, Any]) -> OutputT:
        """Validate output data against the output schema.

        Args:
            data: The data dictionary to validate.

        Returns:
            The validated output model instance.

        Raises:
            OutputValidationError: If validation fails.
        """
        return _validate_with_model(self.output_schema, data, is_input=False)


class AgentContract(Generic[InputT, OutputT]):
    """Encapsulates the input/output contract for an entire agent.

    This contract represents what the agent receives as initial input
    and what it produces as final output (published as event to RabbitMQ).
    """

    def __init__(
        self,
        input_schema: type[InputT],
        output_schema: type[OutputT],
    ) -> None:
        """Initialize the agent contract.

        Args:
            input_schema: The Pydantic model class for agent input validation.
            output_schema: The Pydantic model class for agent output validation.
        """
        self.input_schema = input_schema
        self.output_schema = output_schema

    def validate_input(self, data: dict[str, Any]) -> InputT:
        """Validate input data against the agent input schema.

        Args:
            data: The data dictionary to validate.

        Returns:
            The validated input model instance.

        Raises:
            InputValidationError: If validation fails.
        """
        return _validate_with_model(self.input_schema, data, is_input=True)

    def validate_output(self, data: dict[str, Any]) -> OutputT:
        """Validate output data against the agent output schema.

        Args:
            data: The data dictionary to validate.

        Returns:
            The validated output model instance.

        Raises:
            OutputValidationError: If validation fails.
        """
        return _validate_with_model(self.output_schema, data, is_input=False)
