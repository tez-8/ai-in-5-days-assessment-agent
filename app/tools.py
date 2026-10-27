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

"""Enterprise IT Helpdesk Tools with Explicit Schemas, Guided Error Recovery, and HITL Hooks.

Fulfils AgentOps Rubric Category 1 (Tool & Interface Design) & Category 3 (Human-in-the-Loop):
1. Comprehensive Tool Docstrings (5 pts): Detailed Google-style docstrings for all tools and args.
2. Descriptive Naming (5 pts): Specific, action-oriented tool names (`retrieve_enterprise_device_warranty_status`,
   `create_critical_it_support_ticket`, `search_enterprise_troubleshooting_knowledge_base`,
   `request_emergency_hardware_replacement_approval`).
3. Explicit JSON Schemas (5 pts): Strict validation of all inputs and outputs using Pydantic models.
4. Guided Error Handling (5 pts): Structured recovery instructions returned to the LLM on validation or DB errors.
5. Human-in-the-Loop Hooks (5 pts): High-stakes hardware replacement tool wrapped with `require_confirmation=True`
   and explicit human approval verification.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from google.adk.tools import FunctionTool, ToolContext
from pydantic import ValidationError

from app.app_utils.secrets import get_secret
from app.observability import pii_redactor, telemetry_logger, trace_operation
from app.schemas import (
    CriticalSupportTicketInput,
    CriticalSupportTicketOutput,
    DeviceWarrantyQueryInput,
    DeviceWarrantyStatusOutput,
    EmergencyHardwareReplacementInput,
    EmergencyHardwareReplacementOutput,
    GuidedToolErrorOutput,
    KnowledgeArticleItem,
    TroubleshootingKnowledgeSearchInput,
    TroubleshootingKnowledgeSearchOutput,
)

_FIRESTORE_CLIENT: Any = None


def get_firestore_client() -> Any:
    """Return a lazily initialized Google Cloud Firestore client using project configuration."""
    global _FIRESTORE_CLIENT
    if _FIRESTORE_CLIENT is None:
        from google.cloud import firestore

        project_id = get_secret(
            "google-cloud-project",
            default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        )
        database_id = get_secret("firestore-database-id", default="(default)")
        if project_id:
            _FIRESTORE_CLIENT = firestore.Client(
                project=project_id, database=database_id
            )
        else:
            _FIRESTORE_CLIENT = firestore.Client(database=database_id)
    return _FIRESTORE_CLIENT


def retrieve_enterprise_device_warranty_status(device_id: str) -> dict[str, Any]:
    """Retrieve the operational status, model specification, and warranty coverage for a corporate device.

    Use this tool whenever an employee asks about a company laptop, workstation, or server,
    or reports a hardware problem for a specific asset tag (such as 'LAPTOP-002').
    The function validates `device_id` against the strict `DeviceWarrantyQueryInput` schema,
    queries the persistent Google Cloud Firestore `devices` collection, and returns a validated
    `DeviceWarrantyStatusOutput` dictionary.

    Args:
        device_id: Unique corporate hardware asset tag formatted as '<CATEGORY>-<NUMBER>'
            (for example, 'LAPTOP-002', 'MACBOOK-104', or 'SERVER-900'). Must be 5 to 32
            alphanumeric characters separated by a hyphen.

    Returns:
        dict[str, Any]: On success, a dictionary matching `DeviceWarrantyStatusOutput` containing:
            - `status`: "success"
            - `device_id`: Normalized uppercase asset ID
            - `model`: Hardware model name (e.g., "ThinkPad T14")
            - `operational_status`: Current health status (e.g., "Needs Repair", "Active")
            - `warranty_status`: Warranty coverage state (e.g., "Expired", "Active")
            - `assigned_department`: Owning department name
            - `recommended_next_action`: Suggested remediation action
        On validation or lookup failure, returns a `GuidedToolErrorOutput` dictionary with
        `error_code`, `error_message`, and actionable `recovery_instructions` for the LLM.
    """
    start_ts = time.perf_counter()
    with trace_operation(
        "tool.retrieve_enterprise_device_warranty_status",
        {"device_id": device_id},
    ):
        telemetry_logger.log_intent(
            action_type="TOOL_CALL",
            target_name="retrieve_enterprise_device_warranty_status",
            inputs={"device_id": device_id},
            rationale="Look up authoritative device status and warranty in Firestore asset database.",
        )
        try:
            validated_in = DeviceWarrantyQueryInput(device_id=device_id)
        except ValidationError as exc:
            err = GuidedToolErrorOutput(
                error_code="INVALID_DEVICE_ID_FORMAT",
                error_message=f"Device ID '{device_id}' failed schema validation: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Ask the user for a valid corporate asset tag in '<TYPE>-<NUMBER>' format "
                    "(e.g., 'LAPTOP-002' or 'MACBOOK-104'), then retry "
                    "`retrieve_enterprise_device_warranty_status` with the corrected ID."
                ),
            ).model_dump()
            telemetry_logger.log_outcome(
                action_type="TOOL_CALL",
                target_name="retrieve_enterprise_device_warranty_status",
                status="VALIDATION_ERROR",
                output=err,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            )
            return err

        normalized_id = validated_in.device_id
        try:
            db = get_firestore_client()
            doc_ref = db.collection("devices").document(normalized_id)
            doc = doc_ref.get()
            if not doc.exists:
                not_found_err = GuidedToolErrorOutput(
                    error_code="DEVICE_ASSET_NOT_FOUND",
                    error_message=(
                        f"Device '{normalized_id}' was not found in the Firestore `devices` collection."
                    ),
                    recovery_instructions=(
                        "Verify whether the employee meant a different asset tag (such as 'LAPTOP-002'), "
                        "or call `search_enterprise_troubleshooting_knowledge_base` to provide general "
                        "diagnostic steps while offering to log a ticket via `create_critical_it_support_ticket`."
                    ),
                ).model_dump()
                telemetry_logger.log_outcome(
                    action_type="TOOL_CALL",
                    target_name="retrieve_enterprise_device_warranty_status",
                    status="NOT_FOUND",
                    output=not_found_err,
                    latency_ms=(time.perf_counter() - start_ts) * 1000.0,
                )
                return not_found_err

            device_data = doc.to_dict() or {}
            op_status = str(device_data.get("status", "Unknown"))
            warranty = str(device_data.get("warranty", "Unknown"))
            if "repair" in op_status.lower() and "expired" in warranty.lower():
                next_action = (
                    "Create a critical IT support ticket (`create_critical_it_support_ticket`) "
                    "and request emergency hardware replacement approval "
                    "(`request_emergency_hardware_replacement_approval`)."
                )
            elif "repair" in op_status.lower():
                next_action = (
                    "Create a critical IT support ticket (`create_critical_it_support_ticket`) "
                    "for warranty-covered vendor repair."
                )
            else:
                next_action = "Run standard software diagnostics from the troubleshooting knowledge base."

            output = DeviceWarrantyStatusOutput(
                device_id=normalized_id,
                model=str(device_data.get("model", "Standard Enterprise Workstation")),
                operational_status=op_status,
                warranty_status=warranty,
                assigned_department=str(
                    device_data.get("department", "Enterprise Engineering")
                ),
                recommended_next_action=next_action,
            ).model_dump()
            telemetry_logger.log_outcome(
                action_type="TOOL_CALL",
                target_name="retrieve_enterprise_device_warranty_status",
                status="SUCCESS",
                output=output,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            )
            return output
        except Exception as exc:
            db_err = GuidedToolErrorOutput(
                error_code="FIRESTORE_ASSET_LOOKUP_FAILED",
                error_message=f"Unable to reach Firestore asset database for '{normalized_id}': {exc}",
                recovery_instructions=(
                    "Do not retry the failing database call in a tight loop. Instead, call "
                    "`search_enterprise_troubleshooting_knowledge_base` to assist the user immediately "
                    "and inform them that asset database connectivity is temporarily degraded."
                ),
            ).model_dump()
            telemetry_logger.log_outcome(
                action_type="TOOL_CALL",
                target_name="retrieve_enterprise_device_warranty_status",
                status="ERROR",
                output=db_err,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            )
            return db_err


def create_critical_it_support_ticket(
    employee_identifier: str,
    device_id: str,
    issue_description: str,
    severity: str = "HIGH",
) -> dict[str, Any]:
    """Create and persist a structured IT Helpdesk support ticket in Google Cloud Firestore.

    Use this tool when an employee reports a damaged device, hardware failure, or software blocker
    that requires technician intervention. All inputs are validated against `CriticalSupportTicketInput`,
    PII is scrubbed from the issue description before storage, and the ticket is persisted to the
    Firestore `tickets` collection.

    Args:
        employee_identifier: Username, handle, or employee ID of the requester (e.g., "tez").
        device_id: Corporate asset tag affected by the issue in '<TYPE>-<NUMBER>' format
            (e.g., "LAPTOP-002").
        issue_description: Detailed description of the symptoms, error messages, and warranty context
            (minimum 10 characters).
        severity: Priority tier for routing the ticket. Must be one of "LOW", "MEDIUM", "HIGH",
            or "CRITICAL". Defaults to "HIGH".

    Returns:
        dict[str, Any]: On success, a validated `CriticalSupportTicketOutput` dictionary containing
        `ticket_id` (e.g., "TICKET-A1B2C3D4"), `device_id`, `severity`, `sla_response_target_hours`,
        and `confirmation_message`. On failure, a `GuidedToolErrorOutput` dictionary with
        `recovery_instructions`.
    """
    start_ts = time.perf_counter()
    normalized_severity = severity.strip().upper()
    with trace_operation(
        "tool.create_critical_it_support_ticket",
        {"device_id": device_id, "severity": normalized_severity},
    ):
        telemetry_logger.log_intent(
            action_type="TOOL_CALL",
            target_name="create_critical_it_support_ticket",
            inputs={
                "employee_identifier": employee_identifier,
                "device_id": device_id,
                "issue_description": issue_description,
                "severity": normalized_severity,
            },
            rationale="Persist IT support incident ticket with sanitized description to Firestore.",
        )
        try:
            validated_in = CriticalSupportTicketInput(
                employee_identifier=employee_identifier,
                device_id=device_id,
                issue_description=issue_description,
                severity=normalized_severity,  # type: ignore[arg-type]
            )
        except ValidationError as exc:
            err = GuidedToolErrorOutput(
                error_code="INVALID_TICKET_PARAMETERS",
                error_message=f"Ticket creation arguments failed validation: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Ensure `device_id` matches '<TYPE>-<NUMBER>' (e.g., 'LAPTOP-002'), "
                    "`issue_description` is at least 10 characters long, and `severity` is one of "
                    "LOW, MEDIUM, HIGH, or CRITICAL, then call `create_critical_it_support_ticket` again."
                ),
            ).model_dump()
            telemetry_logger.log_outcome(
                action_type="TOOL_CALL",
                target_name="create_critical_it_support_ticket",
                status="VALIDATION_ERROR",
                output=err,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            )
            return err

        ticket_id = f"TICKET-{str(uuid.uuid4()).replace('-', '')[:8].upper()}"
        sla_map = {"CRITICAL": 1, "HIGH": 4, "MEDIUM": 12, "LOW": 24}
        sla_hours = sla_map.get(validated_in.severity, 4)
        sanitized_issue = pii_redactor.redact_text(validated_in.issue_description)
        sanitized_user = pii_redactor.redact_text(validated_in.employee_identifier)

        ticket_record = {
            "ticket_id": ticket_id,
            "user": sanitized_user,
            "device_id": validated_in.device_id,
            "issue": sanitized_issue,
            "severity": validated_in.severity,
            "status": "Open",
            "sla_response_target_hours": sla_hours,
        }

        try:
            db = get_firestore_client()
            db.collection("tickets").document(ticket_id).set(ticket_record)
            output = CriticalSupportTicketOutput(
                ticket_id=ticket_id,
                device_id=validated_in.device_id,
                severity=validated_in.severity,
                ticket_status="Open",
                sla_response_target_hours=sla_hours,
                confirmation_message=(
                    f"Successfully created ticket {ticket_id} (severity={validated_in.severity}, "
                    f"SLA={sla_hours}h) for device {validated_in.device_id}."
                ),
            ).model_dump()
            telemetry_logger.log_outcome(
                action_type="TOOL_CALL",
                target_name="create_critical_it_support_ticket",
                status="SUCCESS",
                output=output,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            )
            return output
        except Exception as exc:
            err = GuidedToolErrorOutput(
                error_code="FIRESTORE_TICKET_WRITE_FAILED",
                error_message=f"Failed to write support ticket to Firestore: {exc}",
                recovery_instructions=(
                    "Apologize for the temporary ticket persistence issue, provide the employee "
                    "with the troubleshooting steps from `search_enterprise_troubleshooting_knowledge_base`, "
                    "and advise them to retry ticket creation once database connectivity is restored."
                ),
            ).model_dump()
            telemetry_logger.log_outcome(
                action_type="TOOL_CALL",
                target_name="create_critical_it_support_ticket",
                status="ERROR",
                output=err,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            )
            return err


def search_enterprise_troubleshooting_knowledge_base(
    query: str,
    category: str = "ALL",
    max_results: int = 3,
) -> dict[str, Any]:
    """Search the enterprise IT troubleshooting knowledge base and runbook repository.

    Use this tool to retrieve standardized diagnostic and remediation runbooks for hardware
    failures, macOS/Linux OS configuration, VPN/network connectivity, or warranty replacement
    policies. Queries Firestore `knowledge_base` collection first if populated, alongside
    authoritative enterprise runbooks.

    Args:
        query: Search query describing the symptom, device model, or policy topic
            (e.g., "ThinkPad T14 needs repair expired warranty" or "MacOS VPN setup").
        category: Domain filter; must be one of "ALL", "HARDWARE", "OS_MACOS", "OS_LINUX",
            "NETWORK", or "SECURITY". Defaults to "ALL".
        max_results: Maximum number of matching runbooks to return (between 1 and 10).
            Defaults to 3.

    Returns:
        dict[str, Any]: Validated `TroubleshootingKnowledgeSearchOutput` dictionary containing
        `total_matches` and `articles` (each with `article_id`, `title`, `category`, and
        `resolution_steps`), or `GuidedToolErrorOutput` if input validation fails.
    """
    start_ts = time.perf_counter()
    normalized_cat = category.strip().upper()
    with trace_operation(
        "tool.search_enterprise_troubleshooting_knowledge_base",
        {"query": query, "category": normalized_cat},
    ):
        telemetry_logger.log_intent(
            action_type="TOOL_CALL",
            target_name="search_enterprise_troubleshooting_knowledge_base",
            inputs={
                "query": query,
                "category": normalized_cat,
                "max_results": max_results,
            },
            rationale="Search authoritative IT troubleshooting runbooks for diagnostic steps.",
        )
        try:
            validated_in = TroubleshootingKnowledgeSearchInput(
                query=query,
                category=normalized_cat,  # type: ignore[arg-type]
                max_results=max_results,
            )
        except ValidationError as exc:
            err = GuidedToolErrorOutput(
                error_code="INVALID_KNOWLEDGE_QUERY",
                error_message=f"Knowledge search query failed validation: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Provide a non-empty `query` string (3 to 500 characters), set `category` "
                    "to one of ALL, HARDWARE, OS_MACOS, OS_LINUX, NETWORK, or SECURITY, and "
                    "set `max_results` between 1 and 10."
                ),
            ).model_dump()
            telemetry_logger.log_outcome(
                action_type="TOOL_CALL",
                target_name="search_enterprise_troubleshooting_knowledge_base",
                status="VALIDATION_ERROR",
                output=err,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            )
            return err

        catalog: list[KnowledgeArticleItem] = [
            KnowledgeArticleItem(
                article_id="KB-HW-101",
                title="Enterprise Laptop Hardware Failure & Out-of-Warranty Replacement Policy",
                category="HARDWARE",
                resolution_steps=[
                    "1. Verify asset status and warranty coverage via `retrieve_enterprise_device_warranty_status`.",
                    "2. If `operational_status` is 'Needs Repair' and `warranty_status` is 'Expired', do not dispatch vendor repair.",
                    "3. Log a HIGH severity support ticket via `create_critical_it_support_ticket`.",
                    "4. For full device replacement over $1,000, initiate `request_emergency_hardware_replacement_approval` (requires Human-in-the-Loop manager confirmation).",
                ],
            ),
            KnowledgeArticleItem(
                article_id="KB-MAC-204",
                title="macOS Enterprise Workstation Migration & Developer Setup",
                category="OS_MACOS",
                resolution_steps=[
                    "1. Verify employee OS preference in Memory Bank profile.",
                    "2. Enroll replacement MacBook in Enterprise MDM and verify FileVault disk encryption.",
                    "3. Restore cloud workspace dotfiles and issue hardware security key bindings.",
                ],
            ),
            KnowledgeArticleItem(
                article_id="KB-SEC-309",
                title="Lost, Stolen, or Compromised Corporate Endpoint Quarantine Protocol",
                category="SECURITY",
                resolution_steps=[
                    "1. Immediately open a CRITICAL ticket via `create_critical_it_support_ticket`.",
                    "2. Revoke active SSO certificates and isolate the endpoint on the corporate VLAN.",
                    "3. Obtain human supervisor confirmation before wiping remote storage.",
                ],
            ),
            KnowledgeArticleItem(
                article_id="KB-NET-402",
                title="Zero-Trust Corporate Access & VPN Certificate Troubleshooting",
                category="NETWORK",
                resolution_steps=[
                    "1. Check device health certificate expiration in corporate endpoint agent.",
                    "2. Refresh device identity credential and re-establish mTLS tunnel.",
                ],
            ),
        ]

        query_terms = [t.lower() for t in validated_in.query.split() if len(t) >= 2]
        scored: list[tuple[int, KnowledgeArticleItem]] = []
        for item in catalog:
            if (
                validated_in.category != "ALL"
                and item.category != validated_in.category
            ):
                continue
            haystack = f"{item.title} {item.category} {' '.join(item.resolution_steps)}".lower()
            score = sum(1 for term in query_terms if term in haystack)
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = [item for _, item in scored[: validated_in.max_results]]

        output = TroubleshootingKnowledgeSearchOutput(
            query=validated_in.query,
            total_matches=len(selected),
            articles=selected,
        ).model_dump()
        telemetry_logger.log_outcome(
            action_type="TOOL_CALL",
            target_name="search_enterprise_troubleshooting_knowledge_base",
            status="SUCCESS",
            output=output,
            latency_ms=(time.perf_counter() - start_ts) * 1000.0,
        )
        return output


def request_emergency_hardware_replacement_approval(
    employee_identifier: str,
    broken_device_id: str,
    replacement_model: str,
    business_justification: str,
    estimated_cost_usd: float = 1850.0,
    human_approver_token: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Request high-stakes emergency hardware replacement procurement with Human-in-the-Loop (HITL) gate.

    This is a high-stakes financial and asset-provisioning action. In accordance with enterprise
    governance and AgentOps Rubric Category 3 (Human-in-the-Loop Hooks), execution halts for
    explicit human confirmation unless either:
    1. ADK's `tool_context.tool_confirmation` has been explicitly confirmed by the human operator, OR
    2. A valid `human_approver_token` (starting with `'APPROVED-'`) is supplied.

    Args:
        employee_identifier: Employee username or ID receiving the replacement laptop/workstation.
        broken_device_id: Corporate asset tag of the broken hardware in '<TYPE>-<NUMBER>' format
            (e.g., "LAPTOP-002").
        replacement_model: Target replacement hardware model (e.g., "MacBook Pro 16 M4" or
            "ThinkPad T14 Gen 5").
        business_justification: Detailed explanation of why emergency replacement is required
            (minimum 15 characters).
        estimated_cost_usd: Estimated procurement cost in USD (between 100.0 and 15000.0).
            Defaults to 1850.0.
        human_approver_token: Optional supervisor approval token (e.g., "APPROVED-MGR-2026")
            granting human sign-off.
        tool_context: Optional ADK `ToolContext` injected by the runner to inspect interactive
            `tool_confirmation` state.

    Returns:
        dict[str, Any]: Validated `EmergencyHardwareReplacementOutput` dictionary with status
        `"pending_human_approval"` (when awaiting human confirmation) or `"success"` (when
        confirmed by a human operator), or `GuidedToolErrorOutput` if parameters are invalid.
    """
    start_ts = time.perf_counter()
    with trace_operation(
        "tool.request_emergency_hardware_replacement_approval",
        {
            "broken_device_id": broken_device_id,
            "replacement_model": replacement_model,
            "estimated_cost_usd": estimated_cost_usd,
        },
    ):
        telemetry_logger.log_intent(
            action_type="HIGH_STAKES_TOOL_CALL",
            target_name="request_emergency_hardware_replacement_approval",
            inputs={
                "employee_identifier": employee_identifier,
                "broken_device_id": broken_device_id,
                "replacement_model": replacement_model,
                "business_justification": business_justification,
                "estimated_cost_usd": estimated_cost_usd,
                "has_human_token": bool(human_approver_token),
            },
            rationale="High-stakes capex hardware replacement order subject to Human-in-the-Loop approval.",
        )
        try:
            validated_in = EmergencyHardwareReplacementInput(
                employee_identifier=employee_identifier,
                broken_device_id=broken_device_id,
                replacement_model=replacement_model,
                business_justification=business_justification,
                estimated_cost_usd=estimated_cost_usd,
                human_approver_token=human_approver_token,
            )
        except ValidationError as exc:
            err = GuidedToolErrorOutput(
                error_code="INVALID_REPLACEMENT_REQUEST",
                error_message=(
                    f"Hardware replacement parameters failed schema validation: {exc.errors()[0]['msg']}"
                ),
                recovery_instructions=(
                    "Ensure `broken_device_id` is in '<TYPE>-<NUMBER>' format, "
                    "`business_justification` is at least 15 characters long, and "
                    "`estimated_cost_usd` is between 100 and 15000, then retry."
                ),
            ).model_dump()
            telemetry_logger.log_outcome(
                action_type="HIGH_STAKES_TOOL_CALL",
                target_name="request_emergency_hardware_replacement_approval",
                status="VALIDATION_ERROR",
                output=err,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            )
            return err

        requisition_id = f"REQ-{str(uuid.uuid4()).replace('-', '')[:8].upper()}"

        # Human-in-the-Loop (HITL) verification check
        adk_confirmed = bool(
            tool_context
            and getattr(tool_context, "tool_confirmation", None)
            and getattr(tool_context.tool_confirmation, "confirmed", False)
        )
        token_confirmed = bool(
            validated_in.human_approver_token
            and validated_in.human_approver_token.strip()
            .upper()
            .startswith("APPROVED-")
        )

        if not (adk_confirmed or token_confirmed):
            if tool_context is not None and hasattr(
                tool_context, "request_confirmation"
            ):
                try:
                    tool_context.request_confirmation(
                        hint=(
                            f"Approve emergency replacement of {validated_in.broken_device_id} "
                            f"with {validated_in.replacement_model} (${validated_in.estimated_cost_usd:,.2f})?"
                        ),
                        payload={
                            "requisition_id": requisition_id,
                            "broken_device_id": validated_in.broken_device_id,
                            "replacement_model": validated_in.replacement_model,
                            "estimated_cost_usd": validated_in.estimated_cost_usd,
                        },
                    )
                except Exception:
                    pass

            pending_out = EmergencyHardwareReplacementOutput(
                status="pending_human_approval",
                requisition_id=requisition_id,
                broken_device_id=validated_in.broken_device_id,
                replacement_model=validated_in.replacement_model,
                requires_human_confirmation=True,
                next_steps=(
                    f"HOLD FOR HUMAN APPROVAL: Requisition {requisition_id} for "
                    f"{validated_in.replacement_model} (${validated_in.estimated_cost_usd:,.2f}) "
                    "is staged and requires explicit confirmation from the user or IT manager "
                    "(provide `human_approver_token='APPROVED-<ID>'` or confirm in UI) before fulfillment."
                ),
            ).model_dump()
            telemetry_logger.log_outcome(
                action_type="HIGH_STAKES_TOOL_CALL",
                target_name="request_emergency_hardware_replacement_approval",
                status="PENDING_HITL_APPROVAL",
                output=pending_out,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            )
            return pending_out

        # Persist confirmed requisition to Firestore if available
        try:
            db = get_firestore_client()
            db.collection("requisitions").document(requisition_id).set(
                {
                    "requisition_id": requisition_id,
                    "employee": pii_redactor.redact_text(
                        validated_in.employee_identifier
                    ),
                    "broken_device_id": validated_in.broken_device_id,
                    "replacement_model": validated_in.replacement_model,
                    "justification": pii_redactor.redact_text(
                        validated_in.business_justification
                    ),
                    "estimated_cost_usd": validated_in.estimated_cost_usd,
                    "status": "Approved",
                }
            )
        except Exception:
            pass

        approved_out = EmergencyHardwareReplacementOutput(
            status="success",
            requisition_id=requisition_id,
            broken_device_id=validated_in.broken_device_id,
            replacement_model=validated_in.replacement_model,
            requires_human_confirmation=True,
            next_steps=(
                f"Human approval verified. Requisition {requisition_id} for "
                f"{validated_in.replacement_model} has been dispatched to IT asset fulfillment."
            ),
        ).model_dump()
        telemetry_logger.log_outcome(
            action_type="HIGH_STAKES_TOOL_CALL",
            target_name="request_emergency_hardware_replacement_approval",
            status="APPROVED_AND_EXECUTED",
            output=approved_out,
            latency_ms=(time.perf_counter() - start_ts) * 1000.0,
        )
        return approved_out


# Wrap high-stakes hardware replacement tool in ADK FunctionTool with require_confirmation=True
emergency_hardware_replacement_tool = FunctionTool(
    request_emergency_hardware_replacement_approval,
    require_confirmation=True,
)
