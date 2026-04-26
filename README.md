# End-to-End Email Spam Classification

## 1. Project Overview

This project builds an end-to-end machine learning system for **binary email spam classification**.

The system classifies an email into one of two classes:

| Label | Meaning |
|---|---|
| `0` | Ham / Not Spam |
| `1` | Spam |

The project covers the full AI/ML lifecycle:

- Local Medallion data pipeline: Bronze, Silver, Gold
- Exploratory Data Analysis
- Baseline model experiments
- TF-IDF feature extraction
- Logistic Regression model training
- MLflow experiment tracking
- Model evaluation and selection
- FastAPI inference backend
- Streamlit frontend demo
- Prediction logging
- QA testing
- Docker-based deployment

---

## 2. Problem Statement

Email spam is a common problem where users receive unwanted, suspicious, promotional, or fraudulent messages.

The goal of this project is to build a machine learning system that can automatically classify whether an email is **spam** or **not spam** based on its textual content.

The system is designed as a practical AI engineering project, not only a notebook experiment.

---

## 3. Business Objective

The main objective is to create a lightweight, reproducible, and deployable spam classification system.

The final system should allow a user to input:

- Email subject
- Email body

Then the system returns:

- Predicted label: `spam` or `not_spam`
- Spam probability score

Example response:

```json
{
  "label": "spam",
  "spam_probability": 0.94
}
