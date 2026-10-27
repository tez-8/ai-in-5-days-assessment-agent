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

"""Seed Google Cloud Firestore with baseline enterprise hardware assets for verification."""

from __future__ import annotations

import os

from google.cloud import firestore

from app.app_utils.secrets import get_secret
from app.observability import telemetry_logger


def seed_devices(project_id: str | None = None) -> list[str]:
    """Seed sample enterprise devices into the Firestore `devices` collection."""
    resolved_project = project_id or get_secret(
        "google-cloud-project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    )
    db = (
        firestore.Client(project=resolved_project)
        if resolved_project
        else firestore.Client()
    )

    sample_devices = {
        "LAPTOP-002": {
            "model": "ThinkPad T14",
            "status": "Needs Repair",
            "warranty": "Expired",
            "department": "Core Platform Engineering",
        },
        "MACBOOK-104": {
            "model": "MacBook Pro 16 M4",
            "status": "Active",
            "warranty": "Active (Expires 2028-04-30)",
            "department": "AI Research & Enablement",
        },
        "SERVER-900": {
            "model": "PowerEdge R760 GPU Node",
            "status": "Needs Repair",
            "warranty": "Active (Next-Business-Day Onsite)",
            "department": "ML Infrastructure",
        },
    }

    for device_id, payload in sample_devices.items():
        db.collection("devices").document(device_id).set(payload)
        telemetry_logger.log_event(
            "FIRESTORE_SEED",
            f"Seeded device '{device_id}' into Firestore.",
            data={"device_id": device_id, **payload},
        )
    return list(sample_devices.keys())


if __name__ == "__main__":
    seed_devices()
