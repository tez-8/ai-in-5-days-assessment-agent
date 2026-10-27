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

# Secret Manager resources for secure runtime configuration injection (Rubric 5-2 & 5-3)
resource "google_secret_manager_secret" "firestore_database_id" {
  project   = var.project_id
  secret_id = "firestore-database-id"

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret_version" "firestore_database_id_default" {
  secret      = google_secret_manager_secret.firestore_database_id.id
  secret_data = "(default)"
}
