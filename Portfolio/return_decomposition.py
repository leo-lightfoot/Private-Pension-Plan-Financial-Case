"""
Portfolio Return Decomposition
=============================

Analyzes and decomposes portfolio returns by individual asset contributions.
Shows exactly what each asset contributes to the overall portfolio performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ReturnDecomposition:
    def __init__(self, data_file='portfolio_data.csv'):
        """Initialize with historical data."""
        self.data = pd.read_csv(data_file)
        self.data.columns = self.data.columns.str.strip()
        self.data['Dates'] = pd.to_datetime(self.data['Dates'], format='%d-%m-%Y')
        
        # Keep only valid asset columns
        self.assets = [col for col in self.data.columns if col != 'Dates' and not col.startswith('Unnamed')]
        
        # Convert to numeric returns
        for asset in self.assets:
            self.data[asset] = pd.to_numeric(self.data[asset], errors='coerce')
        
        print(f"✅ Loaded data for {len(self.assets)} assets")
        print(f"Assets: {', '.join(self.assets)}")
        print(f"Date range: {self.data['Dates'].min().strftime('%Y-%m-%d')} to {self.data['Dates'].max().strftime('%Y-%m-%d')}")
    
    def calculate_asset_contributions(self, allocations, years=23, n_sims=1000):
        """
        Calculate individual asset contributions to portfolio returns.
        
        Parameters:
        - allocations: dict with asset names and allocation percentages
        - years: number of years to simulate
        - n_sims: number of simulation paths
        """
        print(f"\nAnalyzing asset contributions over {years} years...")
        print(f"Running {n_sims} simulations...")
        
        # Daily returns in decimal
        daily_returns = self.data[self.assets] / 100
        n_days = 252  # trading days in a year
        
        # Store results for each asset
        asset_contributions = {asset: [] for asset in self.assets}
        portfolio_returns = []
        
        for sim in range(n_sims):
            # Track individual asset values
            asset_values = {asset: 1.0 for asset in self.assets}  # Start with 1.0
            portfolio_value = 1.0
            
            for year in range(years):
                # Sample a random block of one year (252 trading days)
                start_idx = np.random.randint(0, len(daily_returns) - n_days)
                block = daily_returns.iloc[start_idx:start_idx+n_days]
                
                # Calculate annual return for each asset
                asset_annual_returns = {}
                for asset in self.assets:
                    asset_daily_returns = block[asset].dropna()
                    if len(asset_daily_returns) > 0:
                        asset_annual_returns[asset] = (1 + asset_daily_returns).prod() - 1
                    else:
                        asset_annual_returns[asset] = 0
                
                # Update individual asset values
                for asset in self.assets:
                    asset_values[asset] *= (1 + asset_annual_returns[asset])
                
                # Calculate portfolio return
                portfolio_return = sum(allocations[asset] * asset_annual_returns[asset] for asset in self.assets)
                portfolio_value *= (1 + portfolio_return)
            
            # Store final values
            portfolio_returns.append(portfolio_value)
            for asset in self.assets:
                asset_contributions[asset].append(asset_values[asset])
        
        return asset_contributions, portfolio_returns
    
    def analyze_contributions(self, allocations, years=23, n_sims=1000):
        """Analyze and display asset contributions."""
        asset_contributions, portfolio_returns = self.calculate_asset_contributions(allocations, years, n_sims)
        
        print("\n" + "="*80)
        print("ASSET CONTRIBUTION ANALYSIS")
        print("="*80)
        
        # Calculate statistics
        portfolio_median = np.median(portfolio_returns)
        portfolio_mean = np.mean(portfolio_returns)
        
        print(f"Portfolio Performance:")
        print(f"  Median Total Return: {(portfolio_median - 1) * 100:.2f}%")
        print(f"  Mean Total Return: {(portfolio_mean - 1) * 100:.2f}%")
        print(f"  Median Annual Return: {(portfolio_median**(1/years) - 1) * 100:.2f}%")
        
        print(f"\nIndividual Asset Contributions:")
        print("-" * 80)
        print(f"{'Asset':<40} {'Allocation':<12} {'Median Return':<15} {'Contribution':<15}")
        print("-" * 80)
        
        total_contribution = 0
        asset_stats = {}
        
        for asset in self.assets:
            if allocations[asset] > 0:  # Only show assets with allocation
                median_return = np.median(asset_contributions[asset])
                annual_return = (median_return**(1/years) - 1) * 100
                contribution = (median_return - 1) * allocations[asset] * 100
                total_contribution += contribution
                
                asset_stats[asset] = {
                    'allocation': allocations[asset],
                    'median_return': median_return,
                    'annual_return': annual_return,
                    'contribution': contribution
                }
                
                print(f"{asset:<40} {allocations[asset]:>10.1%} {annual_return:>13.2f}% {contribution:>13.2f}%")
        
        print("-" * 80)
        print(f"{'TOTAL PORTFOLIO':<40} {'100.0%':<12} {(portfolio_median**(1/years) - 1) * 100:>13.2f}% {total_contribution:>13.2f}%")
        
        return asset_stats, portfolio_returns
    
    def plot_contributions(self, asset_stats, portfolio_returns, years=23):
        """Create visualizations of asset contributions."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Asset Allocation Pie Chart
        allocations = {asset: stats['allocation'] for asset, stats in asset_stats.items() if stats['allocation'] > 0}
        ax1.pie(allocations.values(), labels=allocations.keys(), autopct='%1.1f%%', startangle=90)
        ax1.set_title('Portfolio Allocation', fontsize=14, fontweight='bold')
        
        # 2. Asset Contribution Bar Chart
        assets = list(asset_stats.keys())
        contributions = [asset_stats[asset]['contribution'] for asset in assets]
        colors = ['green' if c > 0 else 'red' for c in contributions]
        
        bars = ax2.bar(range(len(assets)), contributions, color=colors, alpha=0.7)
        ax2.set_title('Asset Contributions to Total Return', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Assets')
        ax2.set_ylabel('Contribution (%)')
        ax2.set_xticks(range(len(assets)))
        ax2.set_xticklabels([asset.replace('_', '\n') for asset in assets], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, contribution in zip(bars, contributions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
                    f'{contribution:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. Asset Annual Returns
        annual_returns = [asset_stats[asset]['annual_return'] for asset in assets]
        bars2 = ax3.bar(range(len(assets)), annual_returns, color='blue', alpha=0.7)
        ax3.set_title('Asset Annual Returns', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Assets')
        ax3.set_ylabel('Annual Return (%)')
        ax3.set_xticks(range(len(assets)))
        ax3.set_xticklabels([asset.replace('_', '\n') for asset in assets], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, return_val in zip(bars2, annual_returns):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{return_val:.1f}%', ha='center', va='bottom')
        
        # 4. Portfolio Value Distribution
        final_values = np.array(portfolio_returns) * 250000  # Convert to actual values
        ax4.hist(final_values, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(np.median(final_values), color='red', linestyle='--', linewidth=2, label=f'Median: €{np.median(final_values):,.0f}')
        ax4.axvline(np.percentile(final_values, 10), color='orange', linestyle='--', linewidth=2, label=f'10th percentile: €{np.percentile(final_values, 10):,.0f}')
        ax4.axvline(np.percentile(final_values, 90), color='green', linestyle='--', linewidth=2, label=f'90th percentile: €{np.percentile(final_values, 90):,.0f}')
        ax4.set_title('Final Portfolio Value Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Final Portfolio Value (€)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('return_decomposition_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_report(self, asset_stats, portfolio_returns, years=23):
        """Create a detailed CSV report of contributions."""
        report_data = []
        
        for asset, stats in asset_stats.items():
            if stats['allocation'] > 0:
                report_data.append({
                    'Asset': asset,
                    'Allocation_Percent': stats['allocation'] * 100,
                    'Median_Total_Return_Percent': (stats['median_return'] - 1) * 100,
                    'Annual_Return_Percent': stats['annual_return'],
                    'Contribution_Percent': stats['contribution']
                })
        
        # Add portfolio summary
        portfolio_median = np.median(portfolio_returns)
        report_data.append({
            'Asset': 'TOTAL_PORTFOLIO',
            'Allocation_Percent': 100.0,
            'Median_Total_Return_Percent': (portfolio_median - 1) * 100,
            'Annual_Return_Percent': (portfolio_median**(1/years) - 1) * 100,
            'Contribution_Percent': sum(stats['contribution'] for stats in asset_stats.values())
        })
        
        df = pd.DataFrame(report_data)
        df.to_csv('asset_contribution_report.csv', index=False)
        print(f"\nDetailed report saved to 'asset_contribution_report.csv'")
        
        return df

def main():
    """Main function to run return decomposition analysis."""
    print("PORTFOLIO RETURN DECOMPOSITION ANALYSIS")
    print("="*50)
    print("Analyzing individual asset contributions to portfolio returns")
    print("="*50)
    
    # Initialize analyzer
    analyzer = ReturnDecomposition()
    
    # Portfolio allocation (same as in the main simulator)
    allocations = {
        'Core_MSCI_World_ETF': 0.50,
        'Core_Eur_govt_bond': 0.05,
        'MSCI_Emerging_Market_ETF': 0.00,
        'Invesco_Physical_Gold_ETF': 0.10,
        'Developed_Markets_Property_Yield_ETF': 0.00,
        'DJGlobal_Real_Estate_ETF': 0.00,
        'Global_Infrastructure_ETF': 0.10,
        'Eur_Core_Corp_Bond_ETF': 0.25,
        'GermanyGovt10.5_Bond_Index': 0.00
    }
    
    # Analyze contributions
    asset_stats, portfolio_returns = analyzer.analyze_contributions(allocations, years=23, n_sims=1000)
    
    # Create visualizations
    analyzer.plot_contributions(asset_stats, portfolio_returns, years=23)
    
    # Create detailed report
    report_df = analyzer.create_detailed_report(asset_stats, portfolio_returns, years=23)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Files generated:")
    print("- return_decomposition_analysis.png")
    print("- asset_contribution_report.csv")

if __name__ == "__main__":
    main()
