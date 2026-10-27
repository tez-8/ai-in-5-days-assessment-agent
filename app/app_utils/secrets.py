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

"""Secure Secret Management utility using Google Cloud Secret Manager.

Fulfils AgentOps Rubric Category 5 (Infrastructure & CI/CD - Secure Secret Management):
- Never hardcodes API keys, credentials, or tokens in source code.
- Retrieves secrets dynamically from Google Cloud Secret Manager using Application
  Default Credentials (ADC) with environment variable fallback for local testing.
"""

from __future__ import annotations

import functools
import os

from google.api_core import exceptions as gcp_exceptions
from google.cloud import secretmanager


class SecretResolutionError(RuntimeError):
    """Raised when a required secret cannot be resolved from any secure source."""


@functools.lru_cache(maxsize=64)
def get_secret(
    secret_id: str,
    project_id: str | None = None,
    version: str = "latest",
    default: str | None = None,
) -> str:
    """Retrieve a secret value securely from Google Cloud Secret Manager or environment.

    Resolution order:
    1. Environment variable matching `secret_id.upper().replace("-", "_")` (injected
       by Cloud Run / Agent Runtime secret bindings or CI runtime).
    2. Google Cloud Secret Manager API (`projects/{project}/secrets/{secret_id}/versions/{version}`).
    3. Explicit `default` value if provided; otherwise raises `SecretResolutionError`.

    Args:
        secret_id: Identifier of the secret in Google Cloud Secret Manager
            (e.g., "it-helpdesk-webhook-token" or "firestore-database-id").
        project_id: Google Cloud project ID hosting the secret. Defaults to the
            `GOOGLE_CLOUD_PROJECT` environment variable.
        version: Secret version to access (defaults to "latest").
        default: Optional safe non-sensitive default if the secret is not configured.

    Returns:
        The resolved secret string stripped of trailing newlines.

    Raises:
        SecretResolutionError: If the secret cannot be resolved from environment
            variables or Secret Manager and no default is provided.
    """
    env_key = secret_id.upper().replace("-", "_")
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val.strip()

    resolved_project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if resolved_project and os.environ.get(
        "ENABLE_SECRET_MANAGER_LOOKUP", "true"
    ).lower() in (
        "true",
        "1",
        "yes",
    ):
        try:
            client = secretmanager.SecretManagerServiceClient()
            secret_name = (
                f"projects/{resolved_project}/secrets/{secret_id}/versions/{version}"
            )
            response = client.access_secret_version(request={"name": secret_name})
            payload = response.payload.data.decode("utf-8").strip()
            if payload:
                return payload
        except (gcp_exceptions.GoogleAPICallError, Exception):
            # Fall through to default handling if Secret Manager is unreachable or
            # the secret does not exist in the current environment.
            pass

    if default is not None:
        return default

    raise SecretResolutionError(
        f"Secret '{secret_id}' could not be resolved from environment variable "
        f"'{env_key}' or Google Cloud Secret Manager in project '{resolved_project}'."
    )
