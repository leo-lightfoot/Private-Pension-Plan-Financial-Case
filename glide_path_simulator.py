"""
Glide Path Portfolio Simulator (minimal edits for config/data consistency)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config_file import (GLIDE_PATH, GLIDE_PATH_BREAKDOWN, SIMULATION,
                   DRAGS, ASSET_CLASSES, CLIENT, TAX_RATES, OUTPUT, DATA_FILES)
from plotting_config import COLOR_SCHEME


class GlidePathSimulator:
    """
    Simulate portfolio performance with dynamic glide path allocation.
    """

    def __init__(self, etf_data, initial_investment=250000):
        """
        Initialize the glide path simulator.

        Args:
            etf_data (pd.DataFrame): Historical daily returns for each ETF
            initial_investment (float): Starting portfolio value in EUR
        """
        self.etf_data = etf_data
        self.initial_investment = initial_investment
        self.current_age = CLIENT['current_age']
        self.retirement_age = CLIENT['retirement_age']
        self.years_to_retirement = self.retirement_age - self.current_age

        # Extract asset parameters from historical data
        self.asset_params = self._estimate_asset_parameters()

        # Create output directory
        os.makedirs(OUTPUT['figures_dir'], exist_ok=True)

    def _estimate_asset_parameters(self):
        """
        Estimate mean and volatility for each asset from historical data.
        Returns daily parameters.
        """
        params = {}
        for asset in self.etf_data.columns:
            # Remove NaN values
            returns = self.etf_data[asset].dropna()

            if len(returns) > 0:
                mu_daily = returns.mean()
                sigma_daily = returns.std()

                # Apply drags (convert annual to daily)
                if asset in ASSET_CLASSES['equity']:
                    mu_daily -= DRAGS['equity_drag'] / SIMULATION['trading_days_per_year']
                elif asset in ASSET_CLASSES['gold']:
                    mu_daily -= DRAGS['gold_drag'] / SIMULATION['trading_days_per_year']
                # Bonds have no drag (or use DRAGS['bond_drag'] if desired)

                params[asset] = (mu_daily, sigma_daily)
            else:
                print(f"Warning: No data for {asset}")
                params[asset] = (0.0, 0.01)  # Default fallback

        return params

    def get_allocation_for_age(self, age):
        """
        Get target portfolio allocation for a given age based on glide path.

        Args:
            age (int): Investor's age

        Returns:
            dict: {asset_name: weight}
        """
        # Find the appropriate glide path bucket
        target_allocation = None
        for (age_min, age_max), allocation in GLIDE_PATH.items():
            if age_min <= age <= age_max:
                target_allocation = allocation
                break

        if target_allocation is None:
            # Default to most conservative if beyond range
            target_allocation = GLIDE_PATH[(66, 68)]

        # Convert high-level allocation to specific ETFs
        detailed_allocation = {}

        # Equity allocation
        equity_pct = target_allocation['equity']
        for etf, weight in GLIDE_PATH_BREAKDOWN['equity'].items():
            detailed_allocation[etf] = equity_pct * weight

        # Bond allocation
        bond_pct = target_allocation['bonds']
        for etf, weight in GLIDE_PATH_BREAKDOWN['bonds'].items():
            detailed_allocation[etf] = bond_pct * weight

        # Gold allocation
        gold_pct = target_allocation['gold']
        for etf, weight in GLIDE_PATH_BREAKDOWN['gold'].items():
            detailed_allocation[etf] = gold_pct * weight

        return detailed_allocation

    def simulate_single_path(self, seed=None):
        """
        Simulate a single portfolio path over the investment horizon.

        Args:
            seed (int, optional): Random seed for reproducibility

        Returns:
            dict: Contains yearly values, allocations, and rebalancing info
        """
        if seed is not None:
            np.random.seed(seed)

        # Track portfolio evolution
        yearly_values = []
        yearly_allocations = []
        rebalancing_log = []

        # Initial allocation at current age
        current_age = self.current_age
        allocation = self.get_allocation_for_age(current_age)

        # Initialize asset values
        asset_values = {asset: self.initial_investment * weight
                       for asset, weight in allocation.items()}

        # Simulate year by year
        for year_idx in range(self.years_to_retirement):
            current_age = self.current_age + year_idx

            # Get target allocation for this age
            target_allocation = self.get_allocation_for_age(current_age)

            # Simulate trading days for this year
            for asset, weight in allocation.items():
                if weight > 0 and asset in self.asset_params:
                    mu_daily, sigma_daily = self.asset_params[asset]

                    # Generate daily returns for one year
                    daily_returns = np.random.normal(mu_daily, sigma_daily, SIMULATION['trading_days_per_year'])

                    # Apply cumulative growth (log-space for numerical stability)
                    growth_factor = np.exp(np.sum(np.log1p(daily_returns)))
                    asset_values[asset] *= growth_factor

            # Calculate total portfolio value after year
            portfolio_value = sum(asset_values.values())
            yearly_values.append(portfolio_value)

            # Store current allocation (before rebalancing)
            current_weights = {asset: value / portfolio_value
                             for asset, value in asset_values.items()}
            yearly_allocations.append(current_weights.copy())

            # Annual rebalancing to target allocation
            # This happens at the end of each year
            old_values = asset_values.copy()
            asset_values = {asset: portfolio_value * target_allocation.get(asset, 0.0)
                          for asset in target_allocation.keys()}

            # Log rebalancing if significant
            total_turnover = sum(abs(asset_values.get(asset, 0) - old_values.get(asset, 0))
                               for asset in set(list(asset_values.keys()) + list(old_values.keys())))
            if total_turnover > portfolio_value * 0.01:  # >1% turnover
                rebalancing_log.append({
                    'year': year_idx + 1,
                    'age': current_age + 1,
                    'portfolio_value': portfolio_value,
                    'turnover': total_turnover,
                    'turnover_pct': total_turnover / portfolio_value
                })

            # Update allocation for next year
            allocation = target_allocation

        return {
            'yearly_values': np.array(yearly_values),
            'yearly_allocations': yearly_allocations,
            'rebalancing_log': rebalancing_log,
            'final_value': yearly_values[-1]
        }

    def simulate_monte_carlo(self, n_simulations=10000):
        """
        Run Monte Carlo simulation with multiple paths.

        Args:
            n_simulations (int): Number of simulation paths

        Returns:
            numpy.ndarray: Array of shape (n_simulations, years)
        """
        print(f"🔄 Running Monte Carlo simulation with {n_simulations:,} paths...")
        print(f"   Glide path: Age {self.current_age} → {self.retirement_age}")
        print(f"   Annual rebalancing enabled")

        results = np.zeros((n_simulations, self.years_to_retirement))

        for sim in range(n_simulations):
            path = self.simulate_single_path(seed=sim)
            results[sim, :] = path['yearly_values']

            # Progress indicator
            if n_simulations >= 10000 and (sim + 1) % (n_simulations // 10) == 0:
                print(f"   Progress: {sim + 1:,}/{n_simulations:,} ({(sim+1)/n_simulations*100:.0f}%)")

        print("✅ Simulation completed!")
        return results

    def apply_capital_gains_tax(self, final_value):
        """
        Apply German capital gains tax on portfolio gains at withdrawal.

        Tax treatment:
        - Equity ETFs: 18.46% (after 30% Teilfreistellung)
        - Bonds/Gold: 26.38% (standard capital gains tax)

        This is a simplified proportional approach.
        """
        total_gains = final_value - self.initial_investment

        final_allocation = self.get_allocation_for_age(self.retirement_age)

        equity_weight = sum(weight for asset, weight in final_allocation.items()
                          if asset in ASSET_CLASSES['equity'])
        bond_weight = sum(weight for asset, weight in final_allocation.items()
                         if asset in ASSET_CLASSES['bonds'])
        gold_weight = sum(weight for asset, weight in final_allocation.items()
                         if asset in ASSET_CLASSES['gold'])

        equity_gains = total_gains * equity_weight
        bond_gains = total_gains * bond_weight
        gold_gains = total_gains * gold_weight

        equity_tax = equity_gains * TAX_RATES['capital_gains_equity']
        bond_tax = bond_gains * TAX_RATES['capital_gains_standard']
        gold_tax = gold_gains * TAX_RATES['capital_gains_standard']

        total_tax = equity_tax + bond_tax + gold_tax
        after_tax_value = final_value - total_tax

        return {
            'pre_tax_value': final_value,
            'total_gains': total_gains,
            'equity_gains': equity_gains,
            'bond_gains': bond_gains,
            'gold_gains': gold_gains,
            'equity_tax': equity_tax,
            'bond_tax': bond_tax,
            'gold_tax': gold_tax,
            'total_tax': total_tax,
            'after_tax_value': after_tax_value,
            'effective_tax_rate': total_tax / total_gains if total_gains > 0 else 0
        }

    def plot_simulation_results(self, results, filename='glide_path_simulation.png'):
        """
        Plot Monte Carlo simulation results.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Top plot: Portfolio value trajectories
        years = range(1, self.years_to_retirement + 1)

        # Plot sample paths
        for i in range(min(100, results.shape[0])):
            ax1.plot(years, results[i], alpha=0.1, color='steelblue', linewidth=0.5)

        # Plot percentiles
        median = np.percentile(results, 50, axis=0)
        p10 = np.percentile(results, 10, axis=0)
        p25 = np.percentile(results, 25, axis=0)
        p75 = np.percentile(results, 75, axis=0)
        p90 = np.percentile(results, 90, axis=0)

        ax1.plot(years, median, color=COLOR_SCHEME['median'], linewidth=2.5, label='Median (50th percentile)')
        ax1.fill_between(years, p10, p90, alpha=0.3, color=COLOR_SCHEME['base'], label='10th-90th percentile')
        ax1.fill_between(years, p25, p75, alpha=0.4, color=COLOR_SCHEME['base'], label='25th-75th percentile')

        # Add initial investment line
        ax1.axhline(y=self.initial_investment, color='gray', linestyle='--',
                   linewidth=1, alpha=0.7, label=f'Initial: €{self.initial_investment:,.0f}')

        ax1.set_xlabel('Years from Now', fontsize=11)
        ax1.set_ylabel('Portfolio Value (€)', fontsize=11)
        ax1.set_title('Portfolio Growth with Glide Path & Annual Rebalancing',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1000:.0f}K'))

        # Bottom plot: Allocation over time
        ages = range(self.current_age, self.retirement_age + 1)
        equity_pcts = []
        bond_pcts = []
        gold_pcts = []

        for age in ages:
            allocation = self.get_allocation_for_age(age)
            equity = sum(w for a, w in allocation.items() if a in ASSET_CLASSES['equity'])
            bonds = sum(w for a, w in allocation.items() if a in ASSET_CLASSES['bonds'])
            gold = sum(w for a, w in allocation.items() if a in ASSET_CLASSES['gold'])
            equity_pcts.append(equity * 100)
            bond_pcts.append(bonds * 100)
            gold_pcts.append(gold * 100)

        ax2.fill_between(range(len(ages)), 0, equity_pcts,
                        alpha=0.7, color=COLOR_SCHEME['equity'], label='Equities')
        ax2.fill_between(range(len(ages)), equity_pcts,
                        [e + b for e, b in zip(equity_pcts, bond_pcts)],
                        alpha=0.7, color=COLOR_SCHEME['bonds'], label='Bonds')
        ax2.fill_between(range(len(ages)),
                        [e + b for e, b in zip(equity_pcts, bond_pcts)],
                        [e + b + g for e, b, g in zip(equity_pcts, bond_pcts, gold_pcts)],
                        alpha=0.7, color=COLOR_SCHEME['gold'], label='Gold')

        ax2.set_xlabel('Age', fontsize=11)
        ax2.set_ylabel('Allocation (%)', fontsize=11)
        ax2.set_title('Glide Path: Asset Allocation Over Time', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(0, len(ages), 3))
        ax2.set_xticklabels([ages[i] for i in range(0, len(ages), 3)])
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        filepath = os.path.join(OUTPUT['figures_dir'], filename)
        plt.savefig(filepath, dpi=OUTPUT['figure_dpi'], bbox_inches='tight')
        plt.close()
        print(f"📊 Simulation plot saved to {filepath}")

    def generate_summary_statistics(self, results):
        """
        Calculate summary statistics from simulation results.
        """
        final_values = results[:, -1]

        # Pre-tax statistics
        stats = {
            'n_simulations': len(final_values),
            'median_final': np.median(final_values),
            'mean_final': np.mean(final_values),
            'p10_final': np.percentile(final_values, 10),
            'p25_final': np.percentile(final_values, 25),
            'p75_final': np.percentile(final_values, 75),
            'p90_final': np.percentile(final_values, 90),
            'median_cagr': (np.median(final_values) / self.initial_investment) ** (1 / self.years_to_retirement) - 1,
            'mean_cagr': (np.mean(final_values) / self.initial_investment) ** (1 / self.years_to_retirement) - 1,
        }

        # After-tax statistics
        tax_analysis = self.apply_capital_gains_tax(stats['median_final'])
        stats['median_after_tax'] = tax_analysis['after_tax_value']
        stats['median_tax_paid'] = tax_analysis['total_tax']
        stats['effective_tax_rate'] = tax_analysis['effective_tax_rate']

        return stats

    def print_summary(self, results):
        """
        Print comprehensive summary of simulation results.
        """
        stats = self.generate_summary_statistics(results)

        print("\n" + "=" * 70)
        print("GLIDE PATH SIMULATION SUMMARY")
        print("=" * 70)

        print(f"\n📊 Simulation Parameters:")
        print(f"   Number of paths: {stats['n_simulations']:,}")
        print(f"   Investment horizon: {self.years_to_retirement} years")
        print(f"   Initial investment: €{self.initial_investment:,.0f}")
        print(f"   Rebalancing: Annual")

        print(f"\n💰 Final Portfolio Value (Pre-Tax):")
        print(f"   Median: €{stats['median_final']:,.0f}")
        print(f"   Mean: €{stats['mean_final']:,.0f}")
        print(f"   10th percentile: €{stats['p10_final']:,.0f}")
        print(f"   90th percentile: €{stats['p90_final']:,.0f}")

        print(f"\n📈 Returns:")
        print(f"   Median CAGR: {stats['median_cagr']:.2%}")
        print(f"   Mean CAGR: {stats['mean_cagr']:.2%}")
        print(f"   Median total growth: {stats['median_final']/self.initial_investment:.2f}x")

        print(f"\n💸 Tax Impact (on Median Outcome):")
        print(f"   Pre-tax value: €{stats['median_final']:,.0f}")
        print(f"   Capital gains tax: €{stats['median_tax_paid']:,.0f}")
        print(f"   After-tax value: €{stats['median_after_tax']:,.0f}")
        print(f"   Effective tax rate: {stats['effective_tax_rate']:.2%}")

        print("\n" + "=" * 70)


def main():
    """Example usage of GlidePathSimulator."""

    # Load historical ETF data from canonical path in config
    print("📂 Loading historical ETF data...")
    data = pd.read_csv(DATA_FILES['etf_returns'], encoding='utf-8-sig')
    try:
        data['Dates'] = pd.to_datetime(data['Dates'], format=DATA_FILES.get('date_format'), dayfirst=True)
    except Exception:
        data['Dates'] = pd.to_datetime(data['Dates'], dayfirst=True, errors='coerce')
    data.set_index('Dates', inplace=True)

    # Convert percentages to decimals
    returns_data = data.apply(pd.to_numeric, errors='coerce') / 100

    # Initialize simulator
    simulator = GlidePathSimulator(returns_data, initial_investment=CLIENT['initial_investment'])

    # Run simulation
    results = simulator.simulate_monte_carlo(n_simulations=SIMULATION['n_simulations'])

    # Generate outputs
    simulator.plot_simulation_results(results)
    simulator.print_summary(results)

    print("\n✅ Glide path simulation completed!")


if __name__ == "__main__":
    main()
