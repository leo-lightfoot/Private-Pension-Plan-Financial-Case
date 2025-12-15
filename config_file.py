"""
Configuration file for Pension Portfolio Optimization Project as per Case

"""

# ============================================================================
# CLIENT PROFILE (single source of truth)
# ============================================================================
CLIENT = {
    'current_age': 45,                    # current age in STARTING_YEAR (2025)
    'retirement_age': 68,                 # planned retirement age
    'initial_investment': 250_000,        # lump sum invested in STARTING_YEAR
    'starting_year': 2025,                # calendar year corresponding to current_age
    # Note: birth_year = starting_year - current_age
}

# ============================================================================
# SALARY & PENSION ASSUMPTIONS (case-driven salary curve)
# ============================================================================
SALARY = {
    # Ground-truth salary curve (case statement):
    # Age : Annual gross income (EUR)
    'salary_by_age': {
        30: 36_000,
        35: 48_000,
        40: 72_000,
        45: 81_600
    },
    # After age 45, salary grows only with inflation (no real wage growth)
    'inflation_rate': 0.02,               # 2% annual inflation
    'age_30_year': 2010,                  # explicit: age 30 occurred in 2010
}

# ============================================================================
# CAREER TIMELINE (centralized assumptions)
# ============================================================================
CAREER = {
    'start_age': 30,                      # career start age
    'start_year': 2010,                   # career start year (from SALARY['age_30_year'])
}

GERMAN_PENSION = {
    'durchschnittsentgelt_2025': 50_493,  # German average earnings (2025)
    'aktueller_rentenwert_2025': 771.84,  # pension value per point (2025)
    'inflation_rate': 0.02,               # pension projection inflation assumption
    'employee_contribution': 0.093,       # employee pension contribution (9.3%)
    'pension_tax_rate': 0.186,            # effective pension tax used in modelling
}

# ============================================================================
# TAX RATES (simplified, documented)
# ============================================================================
TAX_RATES = {
    'income_tax_rate': 0.25,               # simplified effective income tax (25%)
    'unemployment_insurance': 0.026,
    'care_insurance': 0.034,
    'pension_contribution': 0.093,
    'capital_gains_standard': 0.2638,      # standard rate for bond/gold (simplified)
    'capital_gains_equity': 0.1846,        # equity effective rate after Teilfreistellung (simplified)
    'apply_teilfreistellung': True,
}

# ============================================================================
# HEALTHCARE (PKV - step increase every 3 years)
# ============================================================================
PKV = {
    'monthly_cost_age_45': 700,            # EUR/month at age 45 (2025)
    'increase_rate': 0.03,                 # 3% per step
    'increase_frequency_years': 3,         # step every 3 years
}

# ============================================================================
# RETIREMENT GAP ANALYSIS
# ============================================================================
GAP = {
    'replacement_rate': 0.90,             # need 90% of final net salary
    'inflation_rate': SALARY['inflation_rate'],
    'withdrawal_rate': 0.04,              # 4% rule
}

# ============================================================================
# PORTFOLIO ALLOCATION
# ============================================================================
GLIDE_PATH = {
    (45, 50): {'equity': 0.50, 'bonds': 0.40, 'gold': 0.10},
    (51, 55): {'equity': 0.45, 'bonds': 0.45, 'gold': 0.10},
    (56, 60): {'equity': 0.40, 'bonds': 0.50, 'gold': 0.10},
    (61, 65): {'equity': 0.35, 'bonds': 0.55, 'gold': 0.10},
    (66, 68): {'equity': 0.30, 'bonds': 0.60, 'gold': 0.10},
}

GLIDE_PATH_BREAKDOWN = {
    'equity': {
        'Core_MSCI_World_ETF': 0.80,
        'Global_Infrastructure_ETF': 0.20,
    },
    'bonds': {
        'Core_Eur_govt_bond': 0.50,
        'Eur_Core_Corp_Bond_ETF': 0.50,
    },
    'gold': {
        'Invesco_Physical_Gold_ETF': 1.00,
    }
}

# ============================================================================
# SIMULATION PARAMETERS (default reduced to 10k)
# ============================================================================
SIMULATION = {
    'n_simulations': 50_000,
    'n_simulations_stress': 10_000,
    'trading_days_per_year': 252,
}

# ============================================================================
# DRAG FACTORS (single source of truth)
# ============================================================================
DRAGS = {
    'equity_drag': 0.005,
    'gold_drag': 0.003,
    'bond_drag': 0.001,
}

ASSET_CLASSES = {
    'equity': [
        'Core_MSCI_World_ETF',
        'MSCI_Emerging_Market_ETF',
        'Global_Infrastructure_ETF',
        'Developed_Markets_Property_Yield_ETF',
        'DJGlobal_Real_Estate_ETF',
    ],
    'bonds': [
        'Core_Eur_govt_bond',
        'Eur_Core_Corp_Bond_ETF',
        'GermanyGovt10.5_Bond_Index',
    ],
    'gold': [
        'Invesco_Physical_Gold_ETF',
    ]
}

# ============================================================================
# RISK PARAMETERS
# ============================================================================
RISK = {
    'risk_free_rate': 0.0277,
    'var_confidence': 0.95,
    'cvar_confidence': 0.95,
}

# ============================================================================
# PORTFOLIO OPTIMIZATION CONSTRAINTS
# ============================================================================
PORTFOLIO_CONSTRAINTS = {
    'min_observations': 1000,              # minimum data points required for optimization
    'min_return_constraint': 0.05,         # minimum portfolio return (5% annual)
    'max_asset_weight': 1.0,               # maximum weight per asset (100%)
    'allow_short_selling': False,          # no short positions allowed
}

# ============================================================================
# OUTPUT & DATA
# ============================================================================
OUTPUT = {
    'results_dir': 'results',
    'figures_dir': 'results/figures',
    'reports_dir': 'results/reports',
    'figure_dpi': 300,
    'figure_format': 'png',
}

DATA_FILES = {
    'etf_returns': 'data/new_data.csv',
    'date_format': '%d-%m-%Y',
}
