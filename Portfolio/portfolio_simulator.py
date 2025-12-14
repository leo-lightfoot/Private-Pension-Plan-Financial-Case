"""
Portfolio Simulator for Retirement Planning
==========================================

This simulator calculates portfolio returns over time for retirement planning.
The person is 45 years old and will retire at 68, providing 23 years of investment horizon.

Assets available:
1. Core_MSCI_World_ETF
2. Core_Eur_govt_bond
3. MSCI_Emerging_Market_ETF
4. Invesco_Physical_Gold_ETF
5. Developed_Markets_Property_Yield_ETF
6. DJGlobal_Real_Estate_ETF
7. Global_Infrastructure_ETF
8. Eur_Core_Corp_Bond_ETF
9. GermanyGovt10.5_Bond_Index
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioSimulator:
    def __init__(self, data_file='portfolio_data.csv'):
        """Initialize the portfolio simulator with historical data."""
        self.data_file = data_file
        self.data = None
        self.assets = []
        self.annual_returns = {}
        self.volatilities = {}
        self.correlations = None
        self.load_data()
        
    def load_data(self):
        """Load and clean the portfolio data."""
        print("Loading portfolio data...")
        self.data = pd.read_csv(self.data_file)
        
        # Clean column names
        self.data.columns = self.data.columns.str.strip()
        
        # Convert dates
        self.data['Dates'] = pd.to_datetime(self.data['Dates'], format='%d-%m-%Y')
        
        # Get asset columns (exclude Dates and any unnamed columns)
        self.assets = [col for col in self.data.columns if col != 'Dates' and not col.startswith('Unnamed')]
        
        # Convert to numeric, replacing #N/A with NaN
        for asset in self.assets:
            self.data[asset] = pd.to_numeric(self.data[asset], errors='coerce')
        
        # Remove rows where all assets have NaN
        self.data = self.data.dropna(subset=self.assets, how='all')
        
        print(f"Loaded data for {len(self.assets)} assets from {self.data['Dates'].min().strftime('%Y-%m-%d')} to {self.data['Dates'].max().strftime('%Y-%m-%d')}")
        print(f"Assets: {', '.join(self.assets)}")
        
    def calculate_asset_statistics(self):
        """Calculate annual returns and volatilities for each asset."""
        print("\nCalculating asset statistics...")
        
        # Calculate daily returns (data appears to already be in percentage format)
        daily_returns = self.data[self.assets].copy()
        
        # Calculate annual statistics
        for asset in self.assets:
            # Annual return (mean daily return * 252 trading days)
            self.annual_returns[asset] = daily_returns[asset].mean() * 252
            
            # Annual volatility (std daily return * sqrt(252))
            self.volatilities[asset] = daily_returns[asset].std() * np.sqrt(252)
        
        # Calculate correlation matrix
        self.correlations = daily_returns.corr()
        
        print("Asset Statistics (Annual):")
        print("-" * 50)
        for asset in self.assets:
            print(f"{asset:<35}: Return={self.annual_returns[asset]:>8.2f}%, Vol={self.volatilities[asset]:>6.2f}%")
    
    def simulate_portfolio(self, allocations, initial_value=100000, years=23, rebalance_frequency='annual'):
        """
        Simulate portfolio performance over time.
        
        Parameters:
        - allocations: dict with asset names as keys and allocation percentages as values
        - initial_value: starting portfolio value
        - years: number of years to simulate
        - rebalance_frequency: 'annual', 'quarterly', or 'monthly'
        """
        print(f"\nSimulating portfolio with {years} years to retirement...")
        print(f"Initial value: ${initial_value:,.2f}")
        print(f"Rebalancing frequency: {rebalance_frequency}")
        
        # Validate allocations
        total_allocation = sum(allocations.values())
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Allocations must sum to 1.0, got {total_allocation}")
        
        # Calculate daily returns for simulation
        daily_returns = self.data[self.assets].copy()
        
        # Create simulation results
        simulation_results = []
        current_value = initial_value
        current_date = datetime.now()
        
        # Calculate rebalancing frequency in days
        if rebalance_frequency == 'annual':
            rebalance_days = 252
        elif rebalance_frequency == 'quarterly':
            rebalance_days = 63
        elif rebalance_frequency == 'monthly':
            rebalance_days = 21
        else:
            rebalance_days = 252
        
        # Simulate each year
        for year in range(years):
            year_results = {
                'year': year + 1,
                'age': 45 + year + 1,
                'start_value': current_value,
                'end_value': 0,
                'annual_return': 0,
                'monthly_returns': [],
                'monthly_values': []
            }
            
            # Simulate each month in the year
            monthly_value = current_value
            for month in range(12):
                # Sample random daily returns for this month (assuming 21 trading days per month)
                month_returns = []
                for day in range(21):
                    # Randomly sample a historical return for each asset
                    random_idx = np.random.randint(0, len(daily_returns))
                    day_returns = {}
                    for asset in self.assets:
                        if not pd.isna(daily_returns.iloc[random_idx][asset]):
                            day_returns[asset] = daily_returns.iloc[random_idx][asset] / 100  # Convert percentage to decimal
                        else:
                            day_returns[asset] = 0
                    
                    # Calculate portfolio return for this day
                    portfolio_return = sum(allocations[asset] * day_returns[asset] for asset in self.assets)
                    month_returns.append(portfolio_return)
                
                # Apply monthly return
                monthly_return = sum(month_returns)
                monthly_value *= (1 + monthly_return)
                
                year_results['monthly_returns'].append(monthly_return)
                year_results['monthly_values'].append(monthly_value)
            
            # Calculate annual return
            year_results['end_value'] = monthly_value
            year_results['annual_return'] = (monthly_value - current_value) / current_value
            
            simulation_results.append(year_results)
            current_value = monthly_value
            
            # Rebalance if needed
            if (year + 1) % (rebalance_days // 252) == 0:
                # Rebalance to target allocations
                pass  # For now, we assume no rebalancing as per requirements
        
        return simulation_results
    
    def monte_carlo_simulation(self, allocations, initial_value=100000, years=23, num_simulations=1000):
        """Run Monte Carlo simulation to get distribution of outcomes."""
        print(f"\nRunning Monte Carlo simulation with {num_simulations} scenarios...")
        
        all_simulations = []
        final_values = []
        
        for sim in range(num_simulations):
            simulation = self.simulate_portfolio(allocations, initial_value, years)
            all_simulations.append(simulation)
            final_values.append(simulation[-1]['end_value'])
        
        return all_simulations, final_values
    
    def create_visualizations(self, simulation_results, final_values=None):
        """Create comprehensive visualizations of portfolio performance."""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Portfolio Value Over Time
        ax1 = plt.subplot(3, 3, 1)
        years = [result['year'] for result in simulation_results]
        values = [result['end_value'] for result in simulation_results]
        ages = [result['age'] for result in simulation_results]
        
        plt.plot(ages, values, linewidth=2, color='blue')
        plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Age')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Annual Returns
        ax2 = plt.subplot(3, 3, 2)
        annual_returns = [result['annual_return'] * 100 for result in simulation_results]
        plt.bar(years, annual_returns, alpha=0.7, color='green')
        plt.title('Annual Returns', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Annual Return (%)')
        plt.grid(True, alpha=0.3)
        
        # 3. Cumulative Returns
        ax3 = plt.subplot(3, 3, 3)
        cumulative_returns = [(value / simulation_results[0]['start_value'] - 1) * 100 for value in values]
        plt.plot(ages, cumulative_returns, linewidth=2, color='purple')
        plt.title('Cumulative Returns', fontsize=14, fontweight='bold')
        plt.xlabel('Age')
        plt.ylabel('Cumulative Return (%)')
        plt.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Distribution
        ax4 = plt.subplot(3, 3, 4)
        all_monthly_returns = []
        for result in simulation_results:
            all_monthly_returns.extend(result['monthly_returns'])
        plt.hist([r * 100 for r in all_monthly_returns], bins=30, alpha=0.7, color='orange')
        plt.title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Monthly Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 5. Asset Allocation Pie Chart
        ax5 = plt.subplot(3, 3, 5)
        if hasattr(self, 'current_allocations'):
            plt.pie(self.current_allocations.values(), labels=self.current_allocations.keys(), 
                   autopct='%1.1f%%', startangle=90)
            plt.title('Portfolio Allocation', fontsize=14, fontweight='bold')
        
        # 6. Risk-Return Scatter
        ax6 = plt.subplot(3, 3, 6)
        if self.annual_returns and self.volatilities:
            for asset in self.assets:
                plt.scatter(self.volatilities[asset], self.annual_returns[asset], 
                           s=100, alpha=0.7, label=asset)
            plt.xlabel('Volatility (%)')
            plt.ylabel('Annual Return (%)')
            plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
        
        # 7. Monte Carlo Results (if available)
        if final_values:
            ax7 = plt.subplot(3, 3, 7)
            plt.hist(final_values, bins=50, alpha=0.7, color='red')
            plt.axvline(np.mean(final_values), color='black', linestyle='--', 
                       label=f'Mean: ${np.mean(final_values):,.0f}')
            plt.axvline(np.percentile(final_values, 10), color='red', linestyle='--', 
                       label=f'10th percentile: ${np.percentile(final_values, 10):,.0f}')
            plt.axvline(np.percentile(final_values, 90), color='red', linestyle='--', 
                       label=f'90th percentile: ${np.percentile(final_values, 90):,.0f}')
            plt.title('Final Portfolio Value Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Final Portfolio Value ($)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 8. Correlation Heatmap
        ax8 = plt.subplot(3, 3, 8)
        if self.correlations is not None:
            sns.heatmap(self.correlations, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Asset Correlations', fontsize=14, fontweight='bold')
        
        # 9. Retirement Timeline
        ax9 = plt.subplot(3, 3, 9)
        retirement_year = 68
        current_age = 45
        years_to_retirement = retirement_year - current_age
        
        # Create timeline
        timeline_years = list(range(current_age, retirement_year + 1))
        timeline_values = [simulation_results[i]['end_value'] if i < len(simulation_results) else 0 
                          for i in range(len(timeline_years))]
        
        plt.plot(timeline_years, timeline_values, linewidth=3, color='darkblue')
        plt.axvline(x=retirement_year, color='red', linestyle='--', linewidth=2, 
                   label=f'Retirement at {retirement_year}')
        plt.fill_between(timeline_years, timeline_values, alpha=0.3, color='lightblue')
        plt.title('Retirement Timeline', fontsize=14, fontweight='bold')
        plt.xlabel('Age')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig('Portfolio/portfolio_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, simulation_results, final_values=None, allocations=None):
        """Generate a comprehensive portfolio analysis report."""
        print("\n" + "="*80)
        print("PORTFOLIO SIMULATION REPORT")
        print("="*80)
        
        # Basic statistics
        initial_value = simulation_results[0]['start_value']
        final_value = simulation_results[-1]['end_value']
        total_return = (final_value - initial_value) / initial_value
        annualized_return = (final_value / initial_value) ** (1/len(simulation_results)) - 1
        
        print(f"\nPORTFOLIO OVERVIEW:")
        print(f"Initial Value: ${initial_value:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Investment Period: {len(simulation_results)} years")
        
        # Allocation breakdown
        if allocations:
            print(f"\nPORTFOLIO ALLOCATION:")
            for asset, allocation in allocations.items():
                print(f"{asset:<35}: {allocation:>6.1%}")
        
        # Risk metrics
        annual_returns = [result['annual_return'] for result in simulation_results]
        volatility = np.std(annual_returns)
        sharpe_ratio = np.mean(annual_returns) / volatility if volatility > 0 else 0
        
        print(f"\nRISK METRICS:")
        print(f"Annual Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        
        # Best and worst years
        best_year = max(simulation_results, key=lambda x: x['annual_return'])
        worst_year = min(simulation_results, key=lambda x: x['annual_return'])
        
        print(f"\nPERFORMANCE HIGHLIGHTS:")
        print(f"Best Year: {best_year['year']} (Age {best_year['age']}) - {best_year['annual_return']:.2%}")
        print(f"Worst Year: {worst_year['year']} (Age {worst_year['age']}) - {worst_year['annual_return']:.2%}")
        
        # Monte Carlo results
        if final_values:
            print(f"\nMONTE CARLO ANALYSIS ({len(final_values)} simulations):")
            print(f"Mean Final Value: ${np.mean(final_values):,.2f}")
            print(f"Median Final Value: ${np.median(final_values):,.2f}")
            print(f"10th Percentile: ${np.percentile(final_values, 10):,.2f}")
            print(f"90th Percentile: ${np.percentile(final_values, 90):,.2f}")
            print(f"Probability of Loss: {np.mean([v < initial_value for v in final_values]):.1%}")
        
        # Retirement readiness
        retirement_value = final_value
        print(f"\nRETIREMENT READINESS:")
        print(f"Portfolio Value at Retirement (Age 68): ${retirement_value:,.2f}")
        
        # Simple withdrawal analysis (4% rule)
        annual_withdrawal = retirement_value * 0.04
        monthly_withdrawal = annual_withdrawal / 12
        
        print(f"Potential Annual Withdrawal (4% rule): ${annual_withdrawal:,.2f}")
        print(f"Potential Monthly Withdrawal: ${monthly_withdrawal:,.2f}")
        
        print("\n" + "="*80)

def main():
    """Main function to run the portfolio simulator."""
    # Initialize simulator
    simulator = PortfolioSimulator()
    
    # Calculate asset statistics
    simulator.calculate_asset_statistics()
    
    # Example portfolio allocation (user can modify this)
    # This is a balanced portfolio example
    allocations = {
        'Core_MSCI_World_ETF': 0.30,  # 30% Global Equities
        'Core_Eur_govt_bond': 0.15,   # 15% European Government Bonds
        'MSCI_Emerging_Market_ETF': 0.10,  # 10% Emerging Markets
        'Invesco_Physical_Gold_ETF': 0.05,  # 5% Gold
        'Developed_Markets_Property_Yield_ETF': 0.10,  # 10% Real Estate
        'DJGlobal_Real_Estate_ETF': 0.10,  # 10% Global Real Estate
        'Global_Infrastructure_ETF': 0.10,  # 10% Infrastructure
        'Eur_Core_Corp_Bond_ETF': 0.05,  # 5% Corporate Bonds
        'GermanyGovt10.5_Bond_Index': 0.05  # 5% German Government Bonds
    }
    
    # Store allocations for visualization
    simulator.current_allocations = allocations
    
    print("\n" + "="*60)
    print("PORTFOLIO SIMULATOR - RETIREMENT PLANNING")
    print("="*60)
    print("Person: 45 years old")
    print("Retirement Age: 68 years old")
    print("Investment Horizon: 23 years")
    print("="*60)
    
    # Run single simulation
    print("\nRunning single simulation...")
    simulation_results = simulator.simulate_portfolio(allocations, initial_value=100000, years=23)
    
    # Run Monte Carlo simulation
    all_simulations, final_values = simulator.monte_carlo_simulation(allocations, initial_value=100000, years=23, num_simulations=1000)
    
    # Create visualizations
    simulator.create_visualizations(simulation_results, final_values)
    
    # Generate report
    simulator.generate_report(simulation_results, final_values, allocations)
    
    return simulator, simulation_results, final_values

if __name__ == "__main__":
    simulator, results, final_values = main()
