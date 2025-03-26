# CDSS Test Plan

This document defines the set of tests to implement for the CDSS system.
We aim to cover functional, edge-case, integration, and behavioral correctness.

---

## ✅ Core Recommendation Logic

| Test Name | Description | Type | Status |
|----------|-------------|------|--------|
| test_recommendation_structure_correct | Ensure each recommendation is a list of 7 days × N protocols | Unit | ☐ |
| test_unique_days_per_patient | Ensure each patient has unique day assignments | Unit | ☐ |
| test_all_days_assigned | Collapsed `DAYS` across all protocols include 0–6 | Unit | ☐ |
| test_prescriptions_per_day | Each day has exactly N prescriptions | Unit | ☐ |

---

## 🚫 Edge Cases

| Test Name | Description | Type | Status |
|----------|-------------|------|--------|
| test_patient_with_no_sessions | Should return empty recommendation | Unit | ☐ |
| test_patient_with_no_scores | Should return empty recommendation | Unit | ☐ |
| test_malformed_ppf_entries | System handles malformed contrib vector | Unit | ☐ |

---

## ⚖️ Scoring Behavior

| Test Name | Description | Type | Status |
|----------|-------------|------|--------|
| test_alpha_changes_ranking | Changing alpha modifies recs | Behavioral | ☐ |
| test_weights_affect_scores | Different weights yield different scoring | Behavioral | ☐ |

---

## 🔁 Patient Specificity

| Test Name | Description | Type | Status |
|----------|-------------|------|--------|
| test_different_patients_different_recs | Ensure recommendations vary by patient | Behavioral | ☐ |
| test_same_input_same_output | Ensure determinism with fixed input | Behavioral | ☐ |

---

## 🔗 Integration

| Test Name | Description | Type | Status |
|----------|-------------|------|--------|
| test_pipeline_integration | Test DataLoader + Processor + CDSS | Integration | ☐ |
| test_data_loader_with_empty_patient | Handles patients without sessions | Integration | ☐ |

---

## 🧪 Regression

| Test Name | Description | Type | Status |
|----------|-------------|------|--------|
| test_no_duplicate_days | Prevents regressions from old duplicate-day bug | Regression | ☐ |
