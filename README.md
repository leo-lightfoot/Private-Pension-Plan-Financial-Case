# Private Pension Plan Financial Analysis

A comprehensive Python-based financial analysis tool for retirement planning in Germany, specifically designed for a 45-year-old professional investing €250,000 for retirement at age 68.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Methodology](#methodology)
- [Output Files](#output-files)
- [Website](#website)

---

## Overview

This project provides a complete retirement planning analysis framework that:

1. **Calculates German statutory pension** based on salary history and pension points system
2. **Identifies retirement income gaps** between needs and statutory pension
3. **Optimizes portfolio allocation** using Markowitz mean-variance optimization
4. **Simulates portfolio performance** using Monte Carlo methods with dynamic glide path
5. **Performs stress testing** under adverse market scenarios (market crash, stagflation)
6. **Generates comprehensive reports** with visualizations and recommendations

---

## Problem Statement

**Client Profile:**
- Current Age: 45 years (in 2025)
- Retirement Age: 68 years (in 2048)
- Initial Investment: €250,000 (lump sum in 2025)
- Career Start: Age 30 (2010)
- Investment Horizon: 23 years

**Objectives:**
1. Calculate expected German statutory pension at retirement
2. Determine retirement income gap (90% replacement rate of final net salary)
3. Identify optimal portfolio allocation to close the gap
4. Assess portfolio sustainability under various market conditions
5. Provide actionable recommendations for retirement planning

**Key Constraints:**
- Private health insurance (PKV) with 3% increases every 3 years
- German tax system with Teilfreistellung for equities
- No additional contributions (lump sum investment only)

---

## Features

### 1. German Pension Calculator
- Accurate salary projection using historical salary curve
- Pension points (Rentenpunkte) calculation
- Net pension calculation including taxes and PKV

### 2. Retirement Gap Analysis
- Final net salary calculation with all deductions
- Retirement income needs assessment (90% replacement + PKV)
- Annual and monthly gap calculation
- Longevity risk analysis (scenarios to age 95)

### 3. Investment Analysis
- **Markowitz Portfolio Optimization**: Maximize Sharpe ratio with 5% minimum return constraint
- **Risk-Return Metrics**: Calculate returns, volatility, VaR, CVaR, Sharpe ratios
- **Correlation Analysis**: Asset correlation and covariance matrices
- **Asset Performance Comparison**: Individual ETF/asset performance metrics

### 4. Monte Carlo Simulation
- **Glide Path Strategy**: Dynamic allocation that becomes more conservative with age
  - Age 45-50: 50% equity, 40% bonds, 10% gold
  - Age 66-68: 30% equity, 60% bonds, 10% gold
- **50,000 Simulation Paths**: Statistical projections of portfolio outcomes
- **Performance Metrics**: CAGR, volatility, percentile outcomes

### 5. Stress Testing
- **Market Crash Scenario**: 0.5% additional drag + equity shock
- **Stagflation Scenario**: Reduced returns (equities 6%, bonds 2%, gold 5%)
- **IID Normal Simulation**: Asset-by-asset modeling with historical volatilities

### 6. Comprehensive Reporting
- Markdown reports with tables and charts
- High-resolution visualizations (300 DPI)
- Summary statistics and recommendations
- Export to CSV for further analysis

---

## Project Structure

```
Private-Pension-Plan-Financial-Case/
│
├── data/
│   └── new_data.csv                    # Historical ETF returns data
│
├── results/
│   ├── figures/                        # Generated charts and plots
│   └── reports/                        # Analysis reports and CSV exports
│
├── docs/                               # Static website for GitHub Pages
│   ├── index.html                      # Main webpage
│   └── figures/                        # Visualizations for webpage
│
├── Case Files/
│   └── Problem_Case_Statement.png      # Original problem statement
│
├── config_file.py                      # Configuration parameters (single source of truth)
├── pension_calculator.py               # German statutory pension calculations
├── gap_analysis.py                     # Retirement income gap analysis
├── investment_metrics_calculator.py    # ETF performance metrics and analysis
├── portfolio_optimizer.py              # Markowitz portfolio optimization
├── glide_path_simulator.py             # Monte Carlo with dynamic allocation
├── stress_tests.py                     # Stress testing scenarios
├── run_full_analysis.py                # Master script to run complete analysis
│
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
├── LICENSE                             # License information
└── README.md                           # This file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Private-Pension-Plan-Financial-Case
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Usage

### Run Complete Analysis

To execute the entire analysis pipeline:

```bash
python run_full_analysis.py
```

This will:
1. Calculate German statutory pension
2. Analyze retirement income gap
3. Calculate investment metrics for all ETFs
4. Optimize portfolio using Markowitz method
5. Run Monte Carlo simulation with glide path
6. Perform stress testing (market crash & stagflation)
7. Generate comprehensive final report

**Expected Runtime:** ~2-5 minutes (depending on simulation count)

### Run Individual Modules

You can also run individual analysis modules:

```bash
# Pension calculation only
python pension_calculator.py

# Gap analysis only
python gap_analysis.py

# Investment metrics only
python investment_metrics_calculator.py

# Portfolio optimization only
python portfolio_optimizer.py

# Glide path simulation only
python glide_path_simulator.py

# Stress testing only
python stress_tests.py
```

---

## Configuration

All parameters are centralized in `config_file.py`:

### Key Configuration Sections

**Client Profile:**
```python
CLIENT = {
    'current_age': 45,
    'retirement_age': 68,
    'initial_investment': 250_000,
    'starting_year': 2025,
}
```

**Retirement Gap:**
```python
GAP = {
    'replacement_rate': 0.90,      # 90% of final net salary
    'inflation_rate': 0.02,        # 2% annual inflation
    'withdrawal_rate': 0.04,       # 4% rule
}
```

**Glide Path Allocation:**
```python
GLIDE_PATH = {
    (45, 50): {'equity': 0.50, 'bonds': 0.40, 'gold': 0.10},
    (51, 55): {'equity': 0.45, 'bonds': 0.45, 'gold': 0.10},
    # ... etc
}
```

**Simulation Parameters:**
```python
SIMULATION = {
    'n_simulations': 50_000,           # Monte Carlo paths
    'n_simulations_stress': 50_000,    # Stress test paths
    'trading_days_per_year': 252,
}
```

**Tax Rates:**
```python
TAX_RATES = {
    'income_tax_rate': 0.25,           # Simplified effective rate
    'capital_gains_equity': 0.1846,    # After Teilfreistellung
    'capital_gains_standard': 0.2638,  # For bonds/gold
}
```

---

## Methodology

### 1. German Pension Calculation

The statutory pension is calculated using:

```
Pension Points = (Individual Salary / Average Earnings) × Years Worked
Gross Pension = Pension Points × Pension Value (€771.84 in 2025)
Net Pension = Gross Pension - Pension Tax - PKV
```

**Salary Projection:**
- Historical salary data from age 30-45 (case statement)
- Post-45: salary grows only with inflation (2% p.a.)

### 2. Gap Analysis

```
Final Net Salary = Gross Salary - Social Security - Income Tax - PKV
Retirement Needs = (90% × Final Net Salary) + PKV (at retirement)
Annual Gap = Retirement Needs - Net Statutory Pension
```

### 3. Portfolio Optimization (Markowitz)

**Objective:** Maximize Sharpe Ratio

```
Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility

Subject to:
- Σ weights = 1 (fully invested)
- Portfolio Return ≥ 5% (minimum return constraint)
- weights ≥ 0 (no short selling)
```

**Risk-Free Rate:** 2.77% (German government bonds)

### 4. Monte Carlo Simulation

**Glide Path Approach:**
- Dynamic rebalancing each year based on age
- Gradual shift from equity to bonds as retirement approaches
- Maintains 10% gold allocation throughout (diversification)

**Simulation Method:**
- 10,000 independent paths
- Daily returns sampled from historical distributions
- Annual rebalancing
- Accounts for drag factors (equity: 0.5%, bonds: 0.1%, gold: 0.3%)

### 5. Stress Testing

**Market Crash Scenario:**
- Global drag: +0.5% annual
- Equity shock: -3% additional reduction
- Historical volatilities maintained

**Stagflation Scenario:**
- Equity returns: 6% annual (stressed down from historical)
- Bond returns: 2% annual (low-growth environment)
- Gold returns: 5% annual (inflation hedge)
- Historical volatilities maintained

---

## Output Files

### Generated Reports

All outputs are saved to the `results/` directory:

**Reports (`results/reports/`):**
- `pension_projection.csv` - Career salary and pension point accumulation
- `gap_projection.csv` - Retirement gap projection over 25 years
- `investment_metrics_report.md` - Comprehensive ETF performance analysis
- `optimal_portfolio_report.md` - Markowitz optimization results
- `simulation_results.csv` - Raw Monte Carlo simulation data
- `glide_path_summary.txt` - Glide path simulation statistics
- `stress_crash_summary.txt` - Market crash scenario results
- `stress_stagflation_summary.txt` - Stagflation scenario results
- `final_report.md` - Comprehensive summary of all analyses

**Visualizations (`results/figures/`):**
- `correlation_heatmap.png` - Asset correlation matrix
- `risk_return_chart.png` - Risk-return scatter plot
- `optimal_portfolio_weights.png` - Markowitz allocation pie chart
- `efficient_frontier.png` - Efficient frontier curve
- `glide_path_projection.png` - Glide path simulation paths
- `stress_market_crash.png` - Market crash scenario projection
- `stress_stagflation.png` - Stagflation scenario projection

---

## Website

A static website is available to present the analysis results at [docs/index.html](docs/index.html).

### Setup GitHub Pages

1. Copy figures: `results\figures` → `docs\figures` (manually in File Explorer)
2. Push to GitHub
3. Enable GitHub Pages: Settings → Pages → Source: `main` branch, `/docs` folder
4. Your site will be live at: `https://yourusername.github.io/Private-Pension-Plan-Financial-Case/`

---

## Key Insights & Recommendations

The analysis provides answers to:

1. **How much will the statutory pension provide?**
   - Calculated gross and net pension based on salary history

2. **What is the retirement income gap?**
   - Annual and monthly shortfall between needs and pension

3. **What portfolio allocation is optimal?**
   - Markowitz-optimized weights maximizing risk-adjusted returns
   - Represents the "best" portfolio without risk appetite assumptions

4. **What is the expected portfolio value at retirement?**
   - Median, 10th, and 90th percentile outcomes
   - Required CAGR to close the gap

5. **How resilient is the plan to market downturns?**
   - Stress test results showing worst-case scenarios
   - Probability of meeting retirement goals under adverse conditions

6. **What are the key risk factors?**
   - Longevity risk (living beyond expected age)
   - Market risk (sequence of returns)
   - Inflation risk (purchasing power erosion)

---

## Technologies Used

- **Python 3.8+** - Core programming language
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and analysis
- **SciPy** - Optimization algorithms (Markowitz)
- **Matplotlib** - Data visualization and plotting
- **Seaborn** - Statistical visualizations (heatmaps)

---

**Last Updated:** December 2024
