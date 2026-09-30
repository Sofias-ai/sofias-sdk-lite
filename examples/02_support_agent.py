"""Two-node support-triage agent, wired to run against RabbitMQ.

Demonstrates multiple function nodes with explicit routing
(FieldValueStrategy) plus an AgentRunner ready to consume from a real queue.

Test the graph in-process, no broker required:
    pytest tests/integration/test_support_agent.py

Run against RabbitMQ:
    SOFIAS_QUEUE_NAME=support-agent-tasks \\
    RABBITMQ_HOST=localhost RABBITMQ_USERNAME=guest RABBITMQ_PASSWORD=guest \\
    python examples/02_support_agent.py
"""

from __future__ import annotations

import os

from sofias_sdk_lite import (
    AgentBuilder,
    AgentMessage,
    AgentRunner,
    BaseAgentSettings,
    FieldValueStrategy,
    InputContract,
    NodeContract,
    OutputContract,
    RabbitMQConfig,
    RunnerConfig,
)

URGENT_KEYWORDS = ("down", "outage", "urgent", "broken")


class SupportInput(InputContract):
    message: str


class TriageOutput(OutputContract):
    message: str
    urgency: str


class SupportOutput(OutputContract):
    reply: str
    escalated: bool


class SupportSettings(BaseAgentSettings):
    pass


def classify(data: dict, context: dict | None = None) -> dict:
    message = data["message"]
    urgency = "high" if any(word in message.lower() for word in URGENT_KEYWORDS) else "normal"
    return {"message": message, "urgency": urgency}


def escalate(data: dict, context: dict | None = None) -> dict:
    return {"reply": f"Escalated to a human: {data['message']}", "escalated": True}


def respond(data: dict, context: dict | None = None) -> dict:
    return {"reply": f"Thanks for reaching out — we received: {data['message']}", "escalated": False}


def build_support_agent(workflow=None):
    triage_contract = NodeContract(input_schema=SupportInput, output_schema=TriageOutput)
    reply_contract = NodeContract(input_schema=TriageOutput, output_schema=SupportOutput)

    builder = (
        AgentBuilder("support_agent", version="0.1.0")
        .with_settings_class(SupportSettings)
        .with_contract(input_schema=SupportInput, output_schema=SupportOutput)
        .add_function_node("classify", triage_contract, process_fn=classify)
        .add_function_node("escalate", reply_contract, process_fn=escalate)
        .add_function_node("respond", reply_contract, process_fn=respond)
        .add_route(
            "classify",
            FieldValueStrategy(field="urgency", mapping={"high": "escalate", "normal": "respond"}),
        )
        .set_entry_node("classify")
        .set_terminal("escalate")
        .set_terminal("respond")
    )
    if workflow is not None:
        builder = builder.with_response_workflow(workflow)
    return builder.build()


class SupportAgentRunner(AgentRunner):
    settings_class = SupportSettings

    def build_agent(self, settings, workflow):
        return build_support_agent(workflow)

    def prepare_input(self, task, history, role):
        return AgentMessage(content=SupportInput(message=task.content), conversation_id=task.conversation_id)


if __name__ == "__main__":
    SupportAgentRunner(
        RunnerConfig(
            queue=os.environ["SOFIAS_QUEUE_NAME"],
            agent_name="support_agent",
            response_stream=os.environ.get("RABBITMQ_RESPONSE_STREAM", "chat.responses.stream"),
            rabbitmq=RabbitMQConfig(
                host=os.environ.get("RABBITMQ_HOST", "rabbitmq"),
                amqp_port=int(os.environ.get("RABBITMQ_PORT", "5672")),
                stream_port=int(os.environ.get("RABBITMQ_STREAM_PORT", "5552")),
                username=os.environ["RABBITMQ_USERNAME"],
                password=os.environ["RABBITMQ_PASSWORD"],
                virtual_host=os.environ.get("RABBITMQ_VHOST", "/"),
            ),
        )
    ).run()
