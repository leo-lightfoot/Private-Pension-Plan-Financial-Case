"""
Centralized plotting configuration for consistent visualization style across all modules.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Set global matplotlib style parameters
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("deep")

# Color scheme for consistent plotting
COLOR_SCHEME = {
    'equity': '#2E86AB',       # Blue
    'bonds': '#A23B72',        # Purple
    'gold': '#F18F01',         # Orange
    'median': '#06A77D',       # Green
    'percentiles': '#D0D0D0',  # Light gray
    'p10': '#E63946',          # Red
    'p90': '#06A77D',          # Green
    'crash': '#E63946',        # Red
    'stagflation': '#F77F00',  # Dark orange
    'base': '#2E86AB',         # Blue
}

# Plot size presets
PLOT_SIZES = {
    'small': (8, 5),
    'medium': (12, 6),
    'large': (14, 8),
    'square': (10, 10),
}

def apply_common_formatting(ax, title=None, xlabel=None, ylabel=None):
    """Apply common formatting to a matplotlib axis."""
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def format_currency_axis(ax, axis='y'):
    """Format axis to show currency in thousands/millions."""
    from matplotlib.ticker import FuncFormatter

    def currency_formatter(x, pos):
        if abs(x) >= 1_000_000:
            return f'€{x/1_000_000:.1f}M'
        elif abs(x) >= 1_000:
            return f'€{x/1_000:.0f}K'
        else:
            return f'€{x:.0f}'

    if axis == 'y':
        ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(currency_formatter))

def format_percentage_axis(ax, axis='y'):
    """Format axis to show percentages."""
    from matplotlib.ticker import PercentFormatter

    if axis == 'y':
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    else:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
