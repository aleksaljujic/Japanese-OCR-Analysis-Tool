# Japanese OCR Error Analysis Tool

This project is a pipeline for **Japanese text OCR evaluation and error analysis**.

## Overview

The system combines:

- **PaddleOCR** for performing OCR on Japanese text images
- A **GPT-4o–based AI agent** for detailed, line-by-line analysis of OCR errors

The goal is not only to extract text, but to **understand where and why OCR errors occur**.

## OCR Pipeline

- OCR is performed using **PaddleOCR**
- The pipeline supports Japanese writing systems (kanji, hiragana, katakana)
- OCR results can be evaluated using standard metrics (e.g. CER)
- The OCR component runs in an isolated environment due to dependency constraints

## AI Error Analysis

- A **GPT-4o agent** compares the OCR output with ground-truth text
- The agent provides:
  - Line-by-line comparison
  - Identification of incorrect segments
  - Classification of error types (wrong kanji, missing characters, kana confusion, etc.)
  - Explanations of likely OCR failure causes (visual similarity, segmentation issues, font effects)

## Architecture

- **Streamlit** is used as the frontend for image upload, ground-truth input, and result visualization
- **FastAPI** is used to reliably connect components running in separate Python environments
- OCR and AI analysis run in **isolated environments** to avoid dependency conflicts
- Components communicate via HTTP services for stability and reproducibility

## Purpose

This tool is designed for:

- OCR quality evaluation
- Dataset validation
- Error forensics in Japanese document OCR
- Research and development workflows

---

Generated OCR outputs and environment variables are intentionally excluded form version control.
