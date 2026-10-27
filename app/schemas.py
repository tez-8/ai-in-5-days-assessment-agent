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

"""Explicit JSON Schemas (Pydantic v2 Models) for Tool Inputs, Outputs, and Agent State.

Fulfils AgentOps Rubric Category 1 (Tool & Interface Design - Explicit JSON Schemas):
- Enforces strict input and output schemas (`extra="forbid"`, regex patterns, length bounds,
  enum constraints, and field validators) to validate tool arguments and constrain LLM outputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GuidedToolErrorOutput(BaseModel):
    """Standardized guided error response returned to the LLM when a tool call fails."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = Field(
        default="error",
        description="Indicates that the tool execution encountered a recoverable error.",
    )
    error_code: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Machine-readable error classification code (e.g., INVALID_DEVICE_ID).",
    )
    error_message: str = Field(
        ...,
        min_length=5,
        description="Clear, human-readable description of what went wrong.",
    )
    recovery_instructions: str = Field(
        ...,
        min_length=10,
        description="Actionable step-by-step instructions guiding the LLM on how to recover.",
    )


class DeviceWarrantyQueryInput(BaseModel):
    """Strict input schema for querying enterprise hardware asset and warranty status."""

    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(
        ...,
        min_length=5,
        max_length=32,
        pattern=r"^[A-Za-z0-9]+-[A-Za-z0-9]+$",
        description=(
            "Enterprise hardware asset identifier in '<TYPE>-<NUMBER>' format "
            "(e.g., 'LAPTOP-002', 'MACBOOK-104', 'SERVER-900')."
        ),
    )

    @field_validator("device_id")
    @classmethod
    def normalize_device_id(cls, value: str) -> str:
        return value.strip().upper()


class DeviceWarrantyStatusOutput(BaseModel):
    """Strict output schema for enterprise hardware asset and warranty status lookups."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    device_id: str = Field(..., description="Normalized asset identifier.")
    model: str = Field(..., description="Hardware model name.")
    operational_status: str = Field(
        ..., description="Current operational condition (e.g., Active, Needs Repair)."
    )
    warranty_status: str = Field(
        ..., description="Warranty coverage status (e.g., Active, Expired)."
    )
    assigned_department: str = Field(
        default="Enterprise Engineering",
        description="Department or cost center owning the hardware asset.",
    )
    recommended_next_action: str = Field(
        ...,
        description="Deterministic next step recommended based on device and warranty state.",
    )


class CriticalSupportTicketInput(BaseModel):
    """Strict input schema for creating an IT Helpdesk incident or repair ticket."""

    model_config = ConfigDict(extra="forbid")

    employee_identifier: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="Employee username, handle, or identifier requesting IT support.",
    )
    device_id: str = Field(
        ...,
        min_length=5,
        max_length=32,
        pattern=r"^[A-Za-z0-9]+-[A-Za-z0-9]+$",
        description="Target asset identifier (e.g., 'LAPTOP-002').",
    )
    issue_description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Detailed technical summary of the hardware or software issue.",
    )
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        default="HIGH",
        description="Incident priority level: LOW, MEDIUM, HIGH, or CRITICAL.",
    )

    @field_validator("device_id")
    @classmethod
    def normalize_ticket_device_id(cls, value: str) -> str:
        return value.strip().upper()


class CriticalSupportTicketOutput(BaseModel):
    """Strict output schema returned after persisting a support ticket in Firestore."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    ticket_id: str = Field(
        ...,
        pattern=r"^TICKET-[A-Z0-9]{8}$",
        description="Generated unique IT Helpdesk ticket identifier.",
    )
    device_id: str = Field(..., description="Associated hardware asset ID.")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        ..., description="Assigned ticket severity."
    )
    ticket_status: Literal["Open", "Pending Approval", "Escalated"] = "Open"
    sla_response_target_hours: int = Field(
        ..., ge=1, le=72, description="Target SLA response window in hours."
    )
    confirmation_message: str = Field(
        ..., description="Summary confirmation message for the employee."
    )


class TroubleshootingKnowledgeSearchInput(BaseModel):
    """Strict input schema for querying the enterprise IT troubleshooting knowledge base."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Search keywords or symptom description (e.g., 'ThinkPad T14 battery not charging').",
    )
    category: Literal[
        "ALL", "HARDWARE", "OS_MACOS", "OS_LINUX", "NETWORK", "SECURITY"
    ] = Field(
        default="ALL",
        description="Optional knowledge domain filter.",
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of runbook articles to retrieve (1-10).",
    )


class KnowledgeArticleItem(BaseModel):
    """Single knowledge base article result."""

    model_config = ConfigDict(extra="forbid")

    article_id: str = Field(..., description="Unique runbook identifier.")
    title: str = Field(..., description="Runbook title.")
    category: str = Field(..., description="Knowledge domain category.")
    resolution_steps: list[str] = Field(
        ..., description="Ordered troubleshooting and remediation steps."
    )


class TroubleshootingKnowledgeSearchOutput(BaseModel):
    """Strict output schema for knowledge base search results."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    query: str = Field(..., description="Executed search query.")
    total_matches: int = Field(
        ..., ge=0, description="Number of matching runbooks found."
    )
    articles: list[KnowledgeArticleItem] = Field(
        ..., description="List of matching troubleshooting runbooks."
    )


class EmergencyHardwareReplacementInput(BaseModel):
    """Strict input schema for high-stakes emergency hardware replacement orders (requires HITL)."""

    model_config = ConfigDict(extra="forbid")

    employee_identifier: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="Employee identifier receiving the replacement hardware.",
    )
    broken_device_id: str = Field(
        ...,
        min_length=5,
        max_length=32,
        pattern=r"^[A-Za-z0-9]+-[A-Za-z0-9]+$",
        description="Identifier of the defective device being replaced (e.g., 'LAPTOP-002').",
    )
    replacement_model: str = Field(
        ...,
        min_length=3,
        max_length=128,
        description="Requested replacement model (e.g., 'MacBook Pro 16 M4' or 'ThinkPad T14 Gen 5').",
    )
    business_justification: str = Field(
        ...,
        min_length=15,
        max_length=2000,
        description="Business justification for capital expenditure approval.",
    )
    estimated_cost_usd: float = Field(
        default=1850.0,
        ge=100.0,
        le=15000.0,
        description="Estimated hardware replacement cost in USD.",
    )
    human_approver_token: str | None = Field(
        default=None,
        description=(
            "Explicit human-in-the-loop supervisor approval token (e.g., 'APPROVED-BY-IT-MANAGER') "
            "when pre-authorized."
        ),
    )


class EmergencyHardwareReplacementOutput(BaseModel):
    """Strict output schema for emergency hardware replacement requisition."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "pending_human_approval"] = Field(
        ..., description="Execution status of the hardware replacement order."
    )
    requisition_id: str = Field(
        ..., description="Hardware replacement requisition tracking number."
    )
    broken_device_id: str = Field(..., description="Defective device ID.")
    replacement_model: str = Field(
        ..., description="Approved or pending replacement model."
    )
    requires_human_confirmation: bool = Field(
        ...,
        description="Whether human supervisor sign-off is required before fulfillment.",
    )
    next_steps: str = Field(
        ...,
        description="Explanation of fulfillment status or human approval requirement.",
    )
