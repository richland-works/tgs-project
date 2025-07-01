# Models Directory

This directory contains all **fitted models** used by the project.

> ⚠️ **Note:** Due to their large size, these models are **not stored in version control** and are **not bundled** with the project distribution.

## Setup

All required models must be **generated or downloaded at install time**. Ensure your project installation script handles this step appropriately.

## Purpose

These models are typically:

- Machine learning or statistical models trained on project-specific data.
- Serialized using formats like `.pkl`, `.pt`, `.joblib`, or similar.
- Loaded by runtime components of the application for inference or analysis.
- These include the Word2Vec models and tfidf_models
