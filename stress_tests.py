import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from config_file import ASSET_CLASSES

class IIDAssetPortfolioSimulator:
    def __init__(self, returns_data: pd.DataFrame = None, data_file='portfolio_data.csv', output_dir="results"):
        """
        Initialize with historical data and output folder.

        Args:
            returns_data: DataFrame with returns (already in decimal format, indexed by date)
            data_file: Path to CSV file (used if returns_data is None)
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if returns_data is not None:
            # Data provided directly as DataFrame (assumed to be in decimal format)
            # Strip column names to remove any trailing spaces
            returns_data.columns = returns_data.columns.str.strip()

            self.data = returns_data * 100  # Store percentage version for compatibility
            self.assets = list(returns_data.columns)

            # Convert to returns dict (decimals)
            self.asset_data = {}
            for asset in self.assets:
                series = returns_data[asset].dropna()
                self.asset_data[asset] = series

            print(f"✅ Loaded data for {len(self.assets)} assets")
            print(f"Assets: {', '.join(self.assets)}")
        else:
            # Load from CSV file
            self.data = pd.read_csv(data_file)
            self.data.columns = self.data.columns.str.strip()
            self.data['Dates'] = pd.to_datetime(self.data['Dates'], format='mixed', dayfirst=True)

            # Keep only valid asset columns
            self.assets = [col for col in self.data.columns if col != 'Dates' and not col.startswith('Unnamed')]

            # Convert to returns (decimals)
            self.asset_data = {}
            for asset in self.assets:
                series = pd.to_numeric(self.data[asset], errors='coerce').dropna() / 100
                self.asset_data[asset] = series

            print(f"✅ Loaded data for {len(self.assets)} assets")
            print(f"Assets: {', '.join(self.assets)}")

        # Tag assets by type - import from config for consistency
        self.equity_assets = set(ASSET_CLASSES['equity'])
        self.bond_assets = set(ASSET_CLASSES['bonds'])
        self.gold_assets = set(ASSET_CLASSES['gold'])

    def estimate_asset_parameters(self, allocations, global_drag=0.002, stagflation=False):
        """
        Estimate μ and σ for each asset separately (daily stats).
        If stagflation=True, override means with stressed assumptions:
        - Equities ~6% p.a., Bonds ~2% p.a., Gold ~5% p.a.
        """
        params = {}
        daily_global_drag = global_drag / 252

        for asset, w in allocations.items():
            if w > 0:
                rets = self.asset_data[asset]

                if stagflation:
                    if asset in self.equity_assets:
                        mu_daily = 0.06 / 252   # ~6% annual
                        sigma_daily = rets.std()
                    elif asset in self.bond_assets:
                        mu_daily = 0.02 / 252   # ~2% annual
                        sigma_daily = rets.std()
                    elif asset in self.gold_assets:
                        mu_daily = 0.05 / 252   # ~5% annual
                        sigma_daily = rets.std()
                    else:
                        mu_daily = rets.mean()
                        sigma_daily = rets.std()
                else:
                    mu_daily = rets.mean()
                    sigma_daily = rets.std()

                # Apply global drag
                mu_daily -= daily_global_drag

                params[asset] = (mu_daily, sigma_daily, w)
                print(f"📊 {asset}: μ_daily={mu_daily:.5f}, σ_daily={sigma_daily:.5f}, weight={w:.2f}")
        return params

    def simulate_paths(self, allocations, initial_value=250000, years=23, n_sims=1000, 
                       global_drag=0.002, stagflation=False):
        """
        Simulate each asset as i.i.d. Normal (daily), then combine into portfolio.
        stagflation=True applies equity/bond/gold stressed assumptions.
        """
        print(f"🔄 Starting simulation with {n_sims:,} paths...")
        start_time = time.time()
        
        params = self.estimate_asset_parameters(allocations, global_drag=global_drag, stagflation=stagflation)
        results = np.zeros((n_sims, years))
        
        for sim in range(n_sims):
            asset_values = {a: initial_value * w for a, (_, _, w) in params.items()}
            yearly_values = []
            
            for y in range(years):
                for asset, (mu_d, sigma_d, w) in params.items():
                    # simulate 252 daily returns
                    daily_rets = np.random.normal(mu_d, sigma_d, 252)
                    growth_factor = np.prod(1 + daily_rets)
                    asset_values[asset] *= growth_factor
                
                portfolio_value = sum(asset_values.values())
                yearly_values.append(portfolio_value)
            
            results[sim, :] = yearly_values
            
            if n_sims >= 10000 and (sim + 1) % (n_sims // 10) == 0:
                print(f"   Progress: {sim + 1:,}/{n_sims:,} ({(sim + 1)/n_sims*100:.1f}%)")
        
        end_time = time.time()
        print(f"✅ Simulation completed in {end_time - start_time:.2f} seconds")
        print(f"📊 Generated {n_sims:,} simulation paths over {years} years")
        if stagflation:
            print("⚠️ Stagflation Scenario Applied: Equities ~6%, Bonds ~2%, Gold ~5%")
        return results

    def plot_simulation(self, results, years=23, title="Portfolio Projection", filename="simulation.png"):
        """Plot simulation outcomes and save to file."""
        plt.figure(figsize=(12, 6))
        
        for i in range(min(50, results.shape[0])):  # plot up to 50 paths
            plt.plot(range(1, years+1), results[i], alpha=0.2, color='blue')
        
        median = np.percentile(results, 50, axis=0)
        p10 = np.percentile(results, 10, axis=0)
        p90 = np.percentile(results, 90, axis=0)
        
        plt.plot(range(1, years+1), median, color='red', linewidth=2, label='Median')
        plt.fill_between(range(1, years+1), p10, p90, color='blue', alpha=0.2, label='10th–90th Percentile')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Years")
        plt.ylabel("Portfolio Value (€)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"📁 Projection plot saved to {filepath}")

    def plot_annual_return_histogram(self, annual_returns, var_95, filename="annual_returns_hist.png"):
        """Plot histogram of annual returns with VaR cutoff."""
        plt.figure(figsize=(10, 6))
        plt.hist(annual_returns, bins=50, color="skyblue", edgecolor="black", alpha=0.7)

        # VaR line
        plt.axvline(var_95, color="red", linestyle="--", linewidth=2, label=f"VaR 95% = {var_95:.2%}")

        # Shade left tail (CVaR region)
        plt.axvspan(annual_returns.min(), var_95, color="red", alpha=0.2, label="Worst 5% outcomes")

        plt.title("Distribution of Annual Portfolio Returns", fontsize=14, fontweight="bold")
        plt.xlabel("Annual Return")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"📁 Annual returns histogram saved to {filepath}")

    def print_summary(self, results, initial_value=250000, rf_rate=0.0277, filename="summary.txt"):
        """Print and save summary statistics of the simulation (with annual VaR)."""
        final_values = results[:, -1]

        median = np.median(final_values)
        p10 = np.percentile(final_values, 10)
        p90 = np.percentile(final_values, 90)

        cagr_each = (final_values / initial_value) ** (1 / results.shape[1]) - 1
        median_cagr = np.median(cagr_each)
        mean_cagr = np.mean(cagr_each)

        # Annual returns across all paths
        annual_returns = (results[:, 1:] / results[:, :-1]) - 1
        annual_returns = annual_returns.flatten()

        vol = np.std(annual_returns)
        sharpe = (mean_cagr - rf_rate) / vol if vol > 0 else np.nan

        # Annual VaR and CVaR
        var_95 = np.percentile(annual_returns, 5)
        cvar_95 = annual_returns[annual_returns <= var_95].mean()

        summary_str = (
            "\n" + "="*60 +
            "\nASSET-BY-ASSET IID NORMAL SIMULATION SUMMARY" +
            "\n" + "="*60 +
            f"\nNumber of Simulations: {results.shape[0]:,}" +
            f"\nNumber of Years: {results.shape[1]}" +
            f"\nInitial Value: €{initial_value:,.2f}" +
            f"\nMedian Final Value: €{median:,.2f}" +
            f"\n10th Percentile Final Value: €{p10:,.2f}" +
            f"\n90th Percentile Final Value: €{p90:,.2f}" +
            f"\nMedian CAGR: {median_cagr:.2%}" +
            f"\nMean CAGR: {mean_cagr:.2%}" +
            f"\nAnnualized Volatility: {vol:.2%}" +
            f"\nSharpe Ratio (rf={rf_rate:.2%}): {sharpe:.2f}" +
            f"\nAnnual VaR (95%): {var_95:.2%}" +
            f"\nAnnual CVaR (95%): {cvar_95:.2%}" +
            "\n" + "="*60
        )

        print(summary_str)

        filepath = os.path.join(self.output_dir, filename)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(summary_str)
        print(f"📁 Summary saved to {filepath}")

        # Save annual return histogram in same directory as summary
        histogram_filename = os.path.join(os.path.dirname(filename), "annual_returns_hist.png")
        self.plot_annual_return_histogram(annual_returns, var_95, filename=histogram_filename)


def main():
    print("PORTFOLIO SIMULATOR (Adjusted Stagflation Scenario: Equities ~6%)")
    print("="*40)
    print("Person: 45 years old")
    print("Retirement: 68 years old")
    print("Investment horizon: 23 years")
    print("Features: Equities ~6%, Bonds ~2%, Gold ~5%, historical volatilities")
    print("="*40)
    
    simulator = IIDAssetPortfolioSimulator()
    
    allocations = {
        'Core_MSCI_World_ETF': 0.40,
        'Core_Eur_govt_bond': 0.20,
        'MSCI_Emerging_Market_ETF': 0.00,
        'Invesco_Physical_Gold_ETF': 0.10,
        'Developed_Markets_Property_Yield_ETF': 0.00,
        'DJGlobal_Real_Estate_ETF': 0.00,
        'Global_Infrastructure_ETF': 0.10,
        'Eur_Core_Corp_Bond_ETF': 0.20,
        'GermanyGovt10.5_Bond_Index': 0.00
    }
    
    results = simulator.simulate_paths(
        allocations, initial_value=250000, years=23, n_sims=20000, 
        global_drag=0.002, stagflation=True
    )
    simulator.plot_simulation(results, years=23, title="Portfolio Projection – Adjusted Stagflation (Equities ~6%)", filename="simulation.png")
    simulator.print_summary(results, initial_value=250000, filename="summary.txt")


if __name__ == "__main__":
    main()
