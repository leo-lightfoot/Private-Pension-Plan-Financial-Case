"""
Portfolio optimizer (minimal edits for config consistency)

"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
from config_file import DATA_FILES, PORTFOLIO_CONSTRAINTS, SIMULATION

class PortfolioOptimizer:
    """
    Simple Markowitz portfolio optimizer that maximizes Sharpe ratio.

    Uses historical returns to find the optimal portfolio weights that
    maximize the Sharpe ratio (return per unit of risk).
    """

    def __init__(self, returns_data: pd.DataFrame = None, csv_file_path: str = None):
        """
        Initialize the portfolio optimizer.

        Args:
            returns_data: DataFrame with returns (already in decimal format, indexed by date)
            csv_file_path: Path to CSV file (used if returns_data is None)
        """
        self.optimal_weights = None
        self.optimal_portfolio = None

        if returns_data is not None:
            # Data provided directly as DataFrame (decimal format: 0.08 = 8% return)
            self.returns_data = returns_data
            self.csv_file_path = None

            # Filter assets with sufficient data (from config)
            min_obs = PORTFOLIO_CONSTRAINTS['min_observations']
            valid_assets = []
            for col in self.returns_data.columns:
                if self.returns_data[col].count() >= min_obs:
                    valid_assets.append(col)

            self.returns_data = self.returns_data[valid_assets]

            # Calculate annualized mean returns and covariance matrix
            trading_days = SIMULATION['trading_days_per_year']
            self.mean_returns = self.returns_data.mean() * trading_days
            self.cov_matrix = self.returns_data.cov() * trading_days

            print(f"Portfolio optimization data loaded successfully!")
            print(f"Assets included: {len(self.returns_data.columns)} (min {min_obs} observations)")
            if len(self.returns_data.index) > 0:
                print(f"Date range: {self.returns_data.index.min().strftime('%Y-%m-%d')} to {self.returns_data.index.max().strftime('%Y-%m-%d')}")
        else:
            # Load from CSV file
            if csv_file_path is None:
                csv_file_path = DATA_FILES['etf_returns']
            self.csv_file_path = csv_file_path
            self.data = None
            self.returns_data = None
            self.mean_returns = None
            self.cov_matrix = None
            self._load_data()

    def _load_data(self):
        """Load and prepare the data."""
        try:
            # Load the CSV file with encoding to handle BOM
            data = pd.read_csv(self.csv_file_path, encoding='utf-8-sig')

            # Try to parse dates using config date_format; fall back to flexible parsing
            try:
                data['Dates'] = pd.to_datetime(data['Dates'], format=DATA_FILES.get('date_format'), dayfirst=True)
            except Exception:
                data['Dates'] = pd.to_datetime(data['Dates'], dayfirst=True, errors='coerce')

            data.set_index('Dates', inplace=True)

            # Replace missing values and convert to numeric
            data = data.replace('#N/A N/A', np.nan)
            data = data.apply(pd.to_numeric, errors='coerce')
            data = data.dropna(axis=1, how='all')

            # Convert from percentages (CSV format) to decimals (internal standard)
            self.returns_data = data / 100

            # Filter assets with sufficient data (from config)
            min_obs = PORTFOLIO_CONSTRAINTS['min_observations']
            valid_assets = []
            for col in self.returns_data.columns:
                if self.returns_data[col].count() >= min_obs:
                    valid_assets.append(col)

            self.returns_data = self.returns_data[valid_assets]

            # Calculate annualized mean returns and covariance matrix
            trading_days = SIMULATION['trading_days_per_year']
            self.mean_returns = self.returns_data.mean() * trading_days
            self.cov_matrix = self.returns_data.cov() * trading_days

            print(f"Portfolio optimization data loaded successfully!")
            print(f"Assets included: {len(self.returns_data.columns)} (min {min_obs} observations)")
            if len(self.returns_data.index) > 0:
                print(f"Date range: {self.returns_data.index.min().strftime('%Y-%m-%d')} to {self.returns_data.index.max().strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = None
            self.returns_data = None
            self.mean_returns = None
            self.cov_matrix = None
            return None

    def portfolio_stats(self, weights):
        """
        Calculate portfolio statistics for given weights.

        Args:
            weights (array): Portfolio weights

        Returns:
            tuple: (return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        risk_free_rate = 0.0277  # 2.77% risk-free rate
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol

        return portfolio_return, portfolio_vol, sharpe_ratio

    def negative_sharpe_ratio(self, weights):
        """
        Calculate negative Sharpe ratio for optimization (minimize function).

        Args:
            weights (array): Portfolio weights

        Returns:
            float: Negative Sharpe ratio
        """
        return -self.portfolio_stats(weights)[2]

    def optimize_portfolio(self, min_return=None):
        """
        Find the optimal portfolio weights that maximize Sharpe ratio subject to minimum return constraint.

        Args:
            min_return (float): Minimum required annual return (default: from config)

        Returns:
            dict: Optimal portfolio results
        """
        if min_return is None:
            min_return = PORTFOLIO_CONSTRAINTS['min_return_constraint']
        if self.mean_returns is None or self.cov_matrix is None:
            print("No valid returns data available for optimization.")
            return None

        num_assets = len(self.mean_returns)

        # Constraints: weights sum to 1 AND minimum return requirement
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda x: np.sum(x * self.mean_returns) - min_return}  # min return constraint
        ]

        # Bounds: weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Initial guess: equal weights
        initial_guess = np.array([1 / num_assets] * num_assets)

        # Check if minimum return is achievable (compare to max asset return)
        max_possible_return = self.mean_returns.max()
        if min_return > max_possible_return:
            print(f"❌ Minimum return {min_return*100:.1f}% is higher than best single asset return {max_possible_return*100:.1f}%")
            return None

        # Optimize
        result = minimize(
            self.negative_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )

        if result.success:
            self.optimal_weights = result.x
            portfolio_return, portfolio_vol, sharpe_ratio = self.portfolio_stats(self.optimal_weights)

            self.optimal_portfolio = {
                'weights': self.optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'assets': list(self.returns_data.columns)
            }

            print(f"✅ Portfolio optimization completed successfully with {min_return*100:.1f}% minimum return!")
            return self.optimal_portfolio
        else:
            print(f"❌ Optimization failed! Unable to achieve {min_return*100:.1f}% minimum return.")
            return None

    def get_asset_names(self):
        """Get exact asset names from data."""
        if self.optimal_portfolio is None:
            return []
        return list(self.optimal_portfolio['assets'])

    def plot_optimal_portfolio(self, save_path='optimal_portfolio_weights.png'):
        """
        Create a pie chart of optimal portfolio weights.

        Args:
            save_path (str): Path to save the chart
        """
        if self.optimal_portfolio is None:
            print("No optimal portfolio found. Run optimize_portfolio() first.")
            return

        # Filter out very small weights for cleaner visualization
        weights = self.optimal_weights
        assets = self.get_asset_names()

        # Only show assets with >1% allocation
        significant_weights = []
        significant_assets = []
        other_weight = 0

        for i, weight in enumerate(weights):
            if weight > 0.01:  # 1% threshold
                significant_weights.append(weight)
                significant_assets.append(assets[i])
            else:
                other_weight += weight

        if other_weight > 0:
            significant_weights.append(other_weight)
            significant_assets.append('Others')

        # Create pie chart
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(significant_weights)))

        wedges, texts, autotexts = plt.pie(
            significant_weights,
            labels=significant_assets,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )

        # Improve text formatting
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.title('Optimal Portfolio Allocation\n(Maximizes Sharpe Ratio)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Portfolio allocation chart saved to {save_path}")

    def generate_portfolio_report(self, filename='optimal_portfolio_report.md'):
        """
        Generate a markdown report of the optimal portfolio.

        Args:
            filename (str): Name of the markdown file
        """
        if self.optimal_portfolio is None:
            print("No optimal portfolio found. Run optimize_portfolio() first.")
            return

        # Prepare data
        weights = self.optimal_weights
        assets = self.get_asset_names()
        original_assets = self.optimal_portfolio['assets']

        # Create allocation table
        allocation_data = []
        for i, (asset, orig_asset, weight) in enumerate(zip(assets, original_assets, weights)):
            if weight > 0.001:  # Only show meaningful allocations
                individual_return = self.mean_returns[orig_asset] * 100
                individual_vol = np.sqrt(self.cov_matrix.loc[orig_asset, orig_asset]) * 100
                allocation_data.append({
                    'Asset': asset,
                    'Weight (%)': f"{weight*100:.1f}%",
                    'Individual Return (%)': f"{individual_return:.2f}%",
                    'Individual Volatility (%)': f"{individual_vol:.2f}%"
                })

        allocation_df = pd.DataFrame(allocation_data)

        # Create markdown content
        markdown_content = f"""# Optimal Portfolio Report

> **Generated**: {datetime.now().strftime('%B %d, %Y at %H:%M')}

---

## 🎯 Optimal Portfolio Summary

| Metric | Value |
|--------|-------|
| **Expected Annual Return** | {self.optimal_portfolio['expected_return']*100:.2f}% |
| **Annual Volatility** | {self.optimal_portfolio['volatility']*100:.2f}% |
| **Sharpe Ratio** | {self.optimal_portfolio['sharpe_ratio']:.3f} |
| **Number of Assets** | {len([w for w in weights if w > 0.001])} |

---

## 📊 Portfolio Allocation

{allocation_df.to_markdown(index=False)}

### 🔍 Key Allocation Insights

- **Largest Position**: {assets[np.argmax(weights)]} ({weights.max()*100:.1f}%)
- **Most Diversified**: Portfolio includes {len([w for w in weights if w > 0.05])} major positions (>5%)
- **Concentration**: Top 3 positions represent {sum(sorted(weights, reverse=True)[:3])*100:.1f}% of portfolio

---

## 📈 Portfolio Characteristics

### ✅ **Strengths**
- **Optimized Risk-Return**: Portfolio maximizes return per unit of risk
- **Diversification**: Spread across multiple asset classes
- **Mathematical Efficiency**: Based on Modern Portfolio Theory

### ⚠️ **Considerations**
- **Historical Data**: Based on past performance (not guaranteed future results)
- **Rebalancing**: May require periodic adjustments
- **Transaction Costs**: Consider implementation costs

---

## 🎯 Implementation Guide

### **Rebalancing Schedule**
- **Quarterly Review**: Check for significant drift from target weights
- **Annual Rebalance**: Full portfolio rebalancing recommended
- **Threshold**: Rebalance if any asset exceeds ±5% from target

### **Risk Management**
- **Monitor Correlation**: Watch for changing relationships between assets
- **Economic Conditions**: Adjust for major market regime changes
- **Diversification**: Ensure no single position dominates

---

## 📊 Visual Analysis

### Portfolio Allocation Chart
![Optimal Portfolio Weights](optimal_portfolio_weights.png)

---

## 💡 Investment Notes

This optimal portfolio was constructed using:
- **Methodology**: Markowitz Mean-Variance Optimization
- **Objective**: Maximize Sharpe Ratio subject to minimum return constraint
- **Constraints**: Long-only positions (no short selling), Minimum {PORTFOLIO_CONSTRAINTS['min_return_constraint']*100:.0f}% annual return
- **Data Period**: {self.returns_data.index.min().strftime('%Y')} to {self.returns_data.index.max().strftime('%Y')}

**Disclaimer**: Past performance does not guarantee future results. This analysis is for educational purposes only.

---
*This report was generated using the Portfolio Optimizer*
"""

        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"Portfolio report saved to {filename}")


def main():
    """Main function to run the portfolio optimization."""

    print("="*80)
    print("MARKOWITZ PORTFOLIO OPTIMIZATION")
    print("="*80)

    # Step 1: Initialize optimizer
    print("\n1. Loading data...")
    optimizer = PortfolioOptimizer()  # will use DATA_FILES['etf_returns'] by default

    if optimizer.returns_data is None:
        print("❌ Failed to load data.")
        return

    # Step 2: Optimize portfolio (uses min return from config)
    print(f"\n2. Optimizing portfolio with {PORTFOLIO_CONSTRAINTS['min_return_constraint']*100:.1f}% minimum return...")
    optimal_portfolio = optimizer.optimize_portfolio()

    if optimal_portfolio is None:
        print("❌ Portfolio optimization failed.")
        return

    # Step 3: Generate visualizations
    print("\n3. Creating visualizations...")
    optimizer.plot_optimal_portfolio()

    # Step 4: Generate report
    print("\n4. Generating report...")
    optimizer.generate_portfolio_report()

    # Step 5: Display results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("n="*80)

    print(f"\n🎯 Optimal Portfolio Performance:")
    print(f"   Expected Annual Return: {optimal_portfolio['expected_return']*100:.2f}%")
    print(f"   Annual Volatility: {optimal_portfolio['volatility']*100:.2f}%")
    print(f"   Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.3f}")

    print(f"\n📊 Top Allocations:")
    assets = optimizer.get_asset_names()
    weights = optimal_portfolio['weights']

    # Show top 5 allocations
    top_indices = np.argsort(weights)[::-1][:5]
    for i, idx in enumerate(top_indices, 1):
        if weights[idx] > 0.01:
            print(f"   {i}. {assets[idx]}: {weights[idx]*100:.1f}%")

    print("\n✅ Portfolio optimization completed successfully!")
    print("\nFiles generated:")
    print("  • optimal_portfolio_report.md - Complete portfolio analysis")
    print("  • optimal_portfolio_weights.png - Portfolio allocation chart")


if __name__ == "__main__":
    main()
