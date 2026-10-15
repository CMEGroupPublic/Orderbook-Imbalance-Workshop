# CME Level 2 Order Book Depth & Predictive Alpha Pipeline

A modular, security-hardened, multi-agent orchestration and predictive machine learning pipeline designed to process CME Level 2 (Market-by-Price) reconstructed order book data directly from live GCS bucket partitions.

This repository implements two core systems:
1.  **A Multi-Agent Reconstructed L2 Pipeline:** Powered by **LangGraph**, **Gemini 2.5 Pro (via Google Vertex AI)**, and **Plotly** to retrieve, compute, and visualize smoothed touch and deep harmonic imbalances.
2.  **A 10-Minute Predictive Forecasting Model:** Featuring classical **OLS regression baseline solvers** and highly regularized **XGBoost Regressors & Classifiers** to predict future mid-price direction, outputting a high-fidelity out-of-sample visual CSS-only research brief.

---

## Codebase Directory Structure

The project conforms to professional enterprise engineering standards, fully segregating packages and deliverables:

```text
orderbook_imbalance/
├── src/
│   ├── __init__.py            # Python package initialization file
│   ├── utils.py               # Direct GCS parquet data loading, sanitization & metrics math
│   ├── agents.py              # LangGraph multi-agent nodes (retrieval & visualizer)
│   ├── pipeline.py            # LangGraph StateGraph workflow compiler and runner
│   └── predictive_model.py    # Advanced XGBoost & NumPy OLS predictive solver engine
├── config.yaml            # Upgraded centralized paths, default symbols, and parameters (templated for release)
├── run_pipeline.sh        # Command Line Interface (CLI) trigger shell script
├── README.md              # Reworked final project documentation (this file)
├── LICENSE.txt            # BSD-3 license
├── installed_packages.txt # Static virtual environment package checklist
└── reports/               # Centralized output directory for all visual deliverables
```

---

## Dynamic Graph Inputs & Mathematics

### Inputs
*   `symbol` (str): Target CME futures contract symbol (e.g., `CL`, `GC`, `ES`, `NQ`, `BTC`). Checked against zero-trust alphanumeric regex validators.
*   `start_date` / `end_date` (str): Date/time range window to extract (e.g., `2026-08-17` or `2026-08-18 15:30:00`).

---

### Core Mathematical Calculations & Business Meanings

#### 1. Mid-Price (Top-of-Book)
```text
            Best Bid Price₁ + Best Ask Price₁
Mid-Price = ─────────────────────────────────
                            2
```
*   **Business Meaning:** Represents the baseline consensus fair-value of the contract. It serves as the reference point from which price momentum and out-of-sample directional drift are evaluated. Here, Best Bid Price₁ and Best Ask Price₁ represent the best bid and best ask prices.

#### 2. Top-of-Book Imbalance (NOBI L1)
```text
       Bid Size₁ - Ask Size₁
NOBI = ─────────────────────
       Bid Size₁ + Ask Size₁
```
*   **Business Meaning:** Measures immediate high-frequency supply-demand disparity directly at the touch level (Level 1 Bid and Offer sizes). Highly positive values indicate immediate bid pressure, while negative values signal immediate offer-side dominance.

#### 3. L2 Weighted Deep-Book Imbalance (WOBI k=5 levels)
```text
       5                     5
       ∑ (Bid Size_i / i) -  ∑ (Ask Size_i / i)
      i=1                   i=1
WOBI = ────────────────────────────────────────
       5                     5
       ∑ (Bid Size_i / i) +  ∑ (Ask Size_i / i)
      i=1                   i=1
```
*   **Business Meaning:** Captures deeper institutional buying/selling pressure throughout the Market-by-Price (MBP) book layers, where Bid Size_i and Ask Size_i represent the sizes at level i (for k=5 levels).
*   **Spoofing Mitigation:** High-frequency algorithms often use non-bona fide orders (large orders placed deep in the book with no intent of execution) to trick other participants. The **harmonic decay weight (1/i)** systematically scales down the influence of deeper book layers, isolating true liquidity interest.

#### 4. EMA Smoothing
*   **Business Meaning:** A **20-tick Exponential Moving Average (EMA)** is applied to both the NOBI and WOBI series. This filters out transient quote flickering, leaving a clear, readable trend line that highlights persistent institutional pressure.
*   *Quantitative Trade-off:* This introduces a group delay of ~9.5s on 1-second continuous grids. While highly effective for visualization, raw un-smoothed imbalance features are preferred for ultra-short term forecasting.

---

## Advanced 10-Minute Predictive Model Features

#### 1. Target Construction (10-Minute Forward Returns)
```text
                  Mid-Price_(t+600) - Mid-Price_t
Return_(t+600) = ───────────────────────────────── × 10,000  (in bps)
                            Mid-Price_t
```
*   The binary classification target is defined as: `Direction Target = 1 if Return_(t+600) > 0 else 0`, which represents the 10-minute forward directional movement.

#### 2. Rich Engineered Predictor Features
*   **Imbalance & Price Momentum:** Calculated as diff shifts over 5s, 30s, and 60s windows to measure high-frequency decay.
*   **Micro-Price Divergence:** Measures the spread between standard mid-price and size-weighted micro-price, indicating short-term momentum.
*   **Cyclical Time-of-Day transforms:**
    ```text
    time_sin = sin(2π × elapsed_seconds / 86400)
    time_cos = cos(2π × elapsed_seconds / 86400)
    ```
    This resolves the time-extrapolation failure of non-linear tree-based models out-of-sample (where elapsed_seconds is the elapsed seconds of the day).
*   **Pre-Close Window Flag:** Explicitly marks the 15:30:00 (55800s) late-day session where volatility and volume patterns surge prior to CME floor settlement.

---

## Execution Guide

All execution commands must be run within the context of the workspace to ensure correct paths.

### 1. Running the LangGraph Visual Pipeline
The pipeline pulls direct partitioned Parquet logs from GCS, cleans and standardizes timestamps, computes imbalances, invokes Gemini 2.5 Pro for analysis, and saves an interactive, dark-themed HTML report inside the `/reports` directory.

```bash
# Standalone execution
bash ./run_pipeline.sh [symbol] [start_date] [end_date]

# Example - Gold Futures (GC)
bash ./run_pipeline.sh GC 2026-08-17 2026-08-18

# Example - S&P 500 E-mini (ES)
bash ./run_pipeline.sh ES 2026-08-17 2026-08-18
```

---

### 2. Running the Advanced ML Forecasting Engine
The ML engine executes chronological validation (70% Train, 30% Out-of-Sample Test) to predict 10-minute price direction. It runs OLS, highly regularized XGBoost Regressors, and Classifiers, and outputs a dynamic, standalone HTML metric dashboard inside `/reports`.

```bash
# General Syntax
PYTHONPATH=. .venv/bin/python -m src.predictive_model [symbol] [start_date] [end_date]

# Example - S&P 500 E-mini (ES)
PYTHONPATH=. .venv/bin/python -m src.predictive_model ES 2026-08-17 2026-08-18 > reports/research_report.txt

# Example - Digital Asset Bitcoin (BTC)
PYTHONPATH=. .venv/bin/python -m src.predictive_model BTC 2026-08-17 2026-08-18 > reports/research_report_btc.txt
```

## License
This project is licensed under the terms of the BSD 3-Clause License.