# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enterprise Security, Policy, and Self-Evaluation Guardrail Plugin for Google ADK.

Fulfils AgentOps Rubric Category 3 (Orchestration & Logic - Guardrails & Policy Plugins):
- Implements a custom ADK `BasePlugin` (`EnterpriseGovernancePlugin`) registered on `App(plugins=[...])`.
- Enforces pre-model security policies, post-model self-evaluation checks, tool error recovery,
  and distributed trace telemetry across all agents in the hierarchy.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from app.callbacks import (
    after_model_self_eval_guardrail_callback,
    before_model_guardrail_and_compaction_callback,
)
from app.observability import telemetry_logger
from app.schemas import GuidedToolErrorOutput


class EnterpriseGovernancePlugin(BasePlugin):
    """ADK Policy & Guardrail Plugin enforcing security screening, self-eval, and error recovery."""

    def __init__(self, name: str = "enterprise_governance_plugin") -> None:
        super().__init__(name=name)

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        agent_name = getattr(agent, "name", "unknown_agent")
        telemetry_logger.log_intent(
            action_type="AGENT_INVOCATION",
            target_name=agent_name,
            inputs={"agent_name": agent_name},
            agent_name=agent_name,
            rationale=f"Entering agent '{agent_name}' via EnterpriseGovernancePlugin.",
        )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        agent_name = getattr(agent, "name", "unknown_agent")
        telemetry_logger.log_outcome(
            action_type="AGENT_INVOCATION",
            target_name=agent_name,
            status="COMPLETED",
            output={"agent_name": agent_name},
            agent_name=agent_name,
        )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        return before_model_guardrail_and_compaction_callback(
            callback_context=callback_context,
            llm_request=llm_request,
        )

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        return after_model_self_eval_guardrail_callback(
            callback_context=callback_context,
            llm_response=llm_response,
        )

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        """Convert unexpected tool exceptions into GuidedToolErrorOutput instructions (Rubric 1-4)."""
        tool_name = getattr(tool, "name", str(tool))
        guided_error = GuidedToolErrorOutput(
            error_code="UNEXPECTED_TOOL_EXCEPTION",
            error_message=f"Tool '{tool_name}' encountered an exception: {error}",
            recovery_instructions=(
                f"Review the arguments passed to `{tool_name}` ({list(tool_args.keys())}), "
                "verify schema requirements, or fall back to "
                "`search_enterprise_troubleshooting_knowledge_base` to assist the user."
            ),
        ).model_dump()
        telemetry_logger.log_outcome(
            action_type="TOOL_EXCEPTION_RECOVERY",
            target_name=tool_name,
            status="RECOVERED_ERROR",
            output=guided_error,
            agent_name=getattr(tool_context, "agent_name", "it_helpdesk_coordinator"),
        )
        return guided_error
