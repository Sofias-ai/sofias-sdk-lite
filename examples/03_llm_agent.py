"""Single LLM node, no LLM client in sight.

The agent declares *what* the model should do; where the model lives comes
from the environment (or, under ``AgentRunner``, from the per-task settings).

Run against a gateway:
    SOFIAS_LLM_MODEL=default \\
    SOFIAS_LLM_BASE_URL=http://localhost:8080/v1 \\
    SOFIAS_LLM_API_KEY=... \\
    python examples/03_llm_agent.py

Run offline (scripted fake, handy for a first look):
    python examples/03_llm_agent.py --offline
"""

from __future__ import annotations

import asyncio
import sys

from sofias_sdk_lite import (
    AgentBuilder,
    AgentMessage,
    BaseAgentSettings,
    InputContract,
    LLMNodeConfig,
    LLMResponse,
    NodeContract,
    OutputContract,
    default_llm,
)


class Question(InputContract):
    question: str


class Answer(OutputContract):
    answer: str
    confidence: float


class QASettings(BaseAgentSettings):
    """Nothing to add: model_name / router_url / router_api_key are inherited."""


def build_agent():
    contract = NodeContract(input_schema=Question, output_schema=Answer)
    node_config = LLMNodeConfig(
        name="answerer",
        input_contract=Question,
        output_contract=Answer,
        system_prompt="Answer in one sentence and rate your confidence from 0 to 1.",
    )
    return (
        AgentBuilder("qa_agent", version="0.1.0")
        .with_settings_class(QASettings)
        .with_contract(input_schema=Question, output_schema=Answer)
        .add_llm_node("answerer", node_config, contract)  # no with_llm(): resolved at build()
        .set_entry_node("answerer")
        .set_terminal("answerer")
        .build()
    )


class ScriptedLLM:
    """Minimal ``LLMCallable`` for the offline demo."""

    async def invoke(self, prompt, tools=None, messages=None, system_prompt=None,
                     response_format=None, parallel_tool_calls=None, **_):
        return LLMResponse(content='{"answer": "42", "confidence": 0.9}')


async def main(offline: bool) -> None:
    if offline:
        with default_llm(ScriptedLLM()):
            agent = build_agent()
    else:
        agent = build_agent()  # SOFIAS_LLM_* env vars -> GatewayLLM

    response = await agent.execute(AgentMessage(content=Question(question="Meaning of life?")))
    print("status:", response.status)
    print("content:", response.content)
    print("usage:", response.usage)


if __name__ == "__main__":
    asyncio.run(main(offline="--offline" in sys.argv))
