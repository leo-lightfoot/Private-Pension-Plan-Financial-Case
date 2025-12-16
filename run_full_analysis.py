"""
Master Analysis Script - Complete pension portfolio analysis pipeline with stress testing.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime

from config_file import DATA_FILES, CLIENT, OUTPUT, SIMULATION
from pension_calculator import GermanPensionCalculator
from gap_analysis import RetirementGapAnalyzer
from investment_metrics_calculator import InvestmentMetricsCalculator
from portfolio_optimizer import PortfolioOptimizer
from glide_path_simulator import GlidePathSimulator
from stress_tests import IIDAssetPortfolioSimulator


def create_output_directories():
    """Create necessary output directories."""
    os.makedirs(OUTPUT['results_dir'], exist_ok=True)
    os.makedirs(OUTPUT['figures_dir'], exist_ok=True)
    os.makedirs(OUTPUT['reports_dir'], exist_ok=True)
    print("Output directories created")


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def load_etf_data():
    """Load and prepare historical ETF return data."""
    print("Loading historical ETF data...")

    try:
        data = pd.read_csv(DATA_FILES['etf_returns'], encoding='utf-8-sig')
        data.columns = data.columns.str.strip()
        data['Dates'] = pd.to_datetime(data['Dates'], format=DATA_FILES['date_format'], dayfirst=True)
        data.set_index('Dates', inplace=True)

        data = data.replace('#N/A N/A', np.nan)
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.dropna(axis=1, how='all')

        returns_data = data / 100  # Convert to decimals

        print(f"Loaded data for {len(returns_data.columns)} ETFs")
        print(f"Date range: {returns_data.index.min().strftime('%Y-%m-%d')} to {returns_data.index.max().strftime('%Y-%m-%d')}")
        print(f"Total observations: {len(returns_data):,}")

        return returns_data

    except Exception as e:
        print(f"   Error loading data: {e}")
        print(f"\nPlease ensure {DATA_FILES['etf_returns']} exists!")
        sys.exit(1)


def run_pension_analysis():
    """Step 1: Analyze German statutory pension system."""
    print_header("STEP 1: German Statutory Pension Analysis")
    
    calculator = GermanPensionCalculator()
    calculator.print_summary()
    
    # Generate career projection
    career_df = calculator.generate_career_summary()
    output_path = os.path.join(OUTPUT['reports_dir'], 'pension_projection.csv')
    career_df.to_csv(output_path, index=False)
    print(f"\nCareer projection saved to {output_path}")
    
    return calculator


def run_gap_analysis():
    """Step 2: Calculate retirement income gap."""
    print_header("STEP 2: Retirement Income Gap Analysis")
    
    analyzer = RetirementGapAnalyzer()
    analyzer.print_summary()
    
    # Longevity scenarios
    print("\nLongevity Risk Analysis:")
    print("-" * 70)
    longevity = analyzer.generate_longevity_analysis()
    for scenario, total_gap in longevity.items():
        print(f"   {scenario}: €{total_gap:,.0f} cumulative gap")
    
    # Generate gap projection
    gap_projection = analyzer.project_gap_with_inflation(years_in_retirement=25)
    output_path = os.path.join(OUTPUT['reports_dir'], 'gap_projection.csv')
    gap_projection.to_csv(output_path, index=False)
    print(f"\nGap projection saved to {output_path}")
    
    return analyzer


def run_etf_analysis(returns_data):
    """Step 3: Analyze ETF performance metrics."""
    print_header("STEP 3: ETF Performance Analysis")

    # Pass DataFrame directly (no temp CSV needed)
    calculator = InvestmentMetricsCalculator(returns_data=returns_data)
    results = calculator.generate_summary_report()

    # Create visualizations
    calculator.plot_correlation_heatmap(
        save_path=os.path.join(OUTPUT['figures_dir'], 'correlation_heatmap.png')
    )
    calculator.plot_returns_and_volatility(
        save_path=os.path.join(OUTPUT['figures_dir'], 'risk_return_chart.png')
    )

    # Save report
    report_path = os.path.join(OUTPUT['reports_dir'], 'etf_metrics_report.md')
    calculator.save_results_to_markdown(filename=report_path)

    print(f"\nETF analysis saved to {report_path}")

    return results


def run_markowitz_optimization(returns_data):
    """Step 4: Markowitz portfolio optimization."""
    print_header("STEP 4: Markowitz Portfolio Optimization")

    # Pass DataFrame directly (no temp CSV needed)
    optimizer = PortfolioOptimizer(returns_data=returns_data)
    optimal_portfolio = optimizer.optimize_portfolio(min_return=0.05)

    if optimal_portfolio:
        # Create visualizations
        optimizer.plot_optimal_portfolio(
            save_path=os.path.join(OUTPUT['figures_dir'], 'optimal_portfolio_weights.png')
        )

        # Save report
        report_path = os.path.join(OUTPUT['reports_dir'], 'markowitz_optimization.md')
        optimizer.generate_portfolio_report(filename=report_path)

        print(f"\nOptimization results saved to {report_path}")

    return optimal_portfolio


def run_glide_path_simulation(returns_data):
    """Step 5: Monte Carlo simulation with glide path."""
    print_header("STEP 5: Glide Path Monte Carlo Simulation")
    
    simulator = GlidePathSimulator(returns_data, initial_investment=CLIENT['initial_investment'])
    
    # Run simulation
    results = simulator.simulate_monte_carlo(n_simulations=SIMULATION['n_simulations'])
    
    # Generate visualizations and reports
    simulator.plot_simulation_results(
        results,
        filename='glide_path_simulation.png'
    )
    
    simulator.print_summary(results)
    
    # Save raw results
    final_values = results[:, -1]
    results_df = pd.DataFrame({
        'simulation_id': range(len(final_values)),
        'final_value': final_values,
        'cagr': (final_values / CLIENT['initial_investment']) ** (1/simulator.years_to_retirement) - 1
    })
    
    output_path = os.path.join(OUTPUT['reports_dir'], 'simulation_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nSimulation results saved to {output_path}")
    
    return results


def run_stress_tests(returns_data):
    """Step 6: Stress testing scenarios - INTEGRATED FROM stress_returns.py."""
    print_header("STEP 6: Stress Testing")
    
    print("Running stress test scenarios...")
    print("  - Market Crash Scenario (global drag + equity shock)")
    print("  - Stagflation Scenario (reduced returns across assets)")
    
    # Portfolio allocation (static for stress tests)
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

    # --- SCENARIO 1: MARKET CRASH ---
    print("\nScenario 1: Market Crash")
    print("   - Global drag: 0.5% additional annual")
    print("   - Equity returns reduced by 3%")

    # Pass DataFrame directly (no temp CSV needed)
    simulator_crash = IIDAssetPortfolioSimulator(
        returns_data=returns_data,
        output_dir=OUTPUT['results_dir']
    )

    results_crash = simulator_crash.simulate_paths(
        allocations=allocations,
        initial_value=250000,
        years=23,
        n_sims=SIMULATION['n_simulations_stress'],
        global_drag=0.005,
        stagflation=False
    )

    crash_plot_path = os.path.join(OUTPUT['figures_dir'], 'stress_market_crash.png')
    simulator_crash.plot_simulation(
        results_crash,
        years=23,
        title="Portfolio Projection — Market Crash Scenario",
        filename='stress_market_crash.png'
    )

    simulator_crash.print_summary(
        results_crash,
        initial_value=250000,
        filename=os.path.join(OUTPUT['reports_dir'], 'stress_crash_summary.txt')
    )

    print(f"   Crash scenario saved to {crash_plot_path}")

    # --- SCENARIO 2: STAGFLATION ---
    print("\nScenario 2: Stagflation")
    print("   - Equity returns: 6% annual (stressed down)")
    print("   - Bond returns: 2% annual (stressed down)")
    print("   - Gold returns: 5% annual")
        
    # Pass DataFrame directly (no temp CSV needed)
    simulator_stag = IIDAssetPortfolioSimulator(
        returns_data=returns_data,
        output_dir=OUTPUT['results_dir']
    )

    results_stag = simulator_stag.simulate_paths(
        allocations=allocations,
        initial_value=250000,
        years=23,
        n_sims=SIMULATION['n_simulations_stress'],
        global_drag=0.005,
        stagflation=True  # This triggers the stagflation scenario
    )

    stag_plot_path = os.path.join(OUTPUT['figures_dir'], 'stress_stagflation.png')
    simulator_stag.plot_simulation(
        results_stag,
        years=23,
        title="Portfolio Projection — Stagflation Scenario",
        filename='stress_stagflation.png'
    )

    simulator_stag.print_summary(
        results_stag,
        initial_value=250000,
        filename=os.path.join(OUTPUT['reports_dir'], 'stress_stagflation_summary.txt')
    )

    print(f"   Stagflation scenario saved to {stag_plot_path}")
        
    print("\nStress testing completed!")
    print(f"   Results saved to {OUTPUT['figures_dir']}/")


def generate_final_report(pension_calc, gap_analyzer, simulation_results):
    """Step 7: Generate comprehensive final report."""
    print_header("STEP 7: Generating Final Report")
    
    # Calculate key metrics
    gap_analysis = gap_analyzer.calculate_pension_gap()
    portfolio_req = gap_analyzer.calculate_required_portfolio_value()
    
    final_values = simulation_results[:, -1]
    median_final = np.median(final_values)
    p10_final = np.percentile(final_values, 10)
    p90_final = np.percentile(final_values, 90)
    
    success_rate = np.mean(final_values >= portfolio_req['required_portfolio_4pct']) * 100
    
    # Create summary report
    report = f"""# Private Pension Portfolio - Final Report

**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}

---

## Executive Summary

### Client Profile
- **Age:** {CLIENT['current_age']} years old (retiring at {CLIENT['retirement_age']})
- **Investment Horizon:** {CLIENT['retirement_age'] - CLIENT['current_age']} years
- **Initial Capital:** €{CLIENT['initial_investment']:,}

### Retirement Goals
- **Annual Income Needed:** €{gap_analysis['retirement_needs']:,.0f}
- **Statutory Pension:** €{gap_analysis['net_statutory_pension']:,.0f}
- **Annual Gap:** €{gap_analysis['annual_gap']:,.0f}
- **Required Portfolio (4% rule):** €{portfolio_req['required_portfolio_4pct']:,.0f}

### Projected Outcomes
- **Median Final Value:** €{median_final:,.0f}
- **10th Percentile:** €{p10_final:,.0f}
- **90th Percentile:** €{p90_final:,.0f}
- **Success Rate:** {success_rate:.1f}% (meeting €{portfolio_req['required_portfolio_4pct']:,.0f} target)

---

## Investment Strategy

### Asset Allocation (Glide Path)

**Starting Allocation (Age 45):**
- 50% Equities (40% MSCI World + 10% Infrastructure)
- 40% Bonds (20% Euro Govt + 20% Euro Corp)
- 10% Gold

**Ending Allocation (Age 68):**
- 30% Equities
- 60% Bonds
- 10% Gold

### Key Features
- Annual rebalancing
- Low-cost ETFs (avg 0.23% TER)
- Tax-optimized (German Teilfreistellung)
- Diversified across global markets

---

## Risk Assessment

### Probability Analysis
- **Meeting Target (€{portfolio_req['required_portfolio_4pct']:,.0f}):** {success_rate:.1f}%
- **Exceeding Target by 20%:** {np.mean(final_values >= portfolio_req['required_portfolio_4pct'] * 1.2) * 100:.1f}%
- **Falling Short by 20%:** {np.mean(final_values <= portfolio_req['required_portfolio_4pct'] * 0.8) * 100:.1f}%

### Downside Protection
- **10th Percentile Outcome:** €{p10_final:,.0f}
- **Minimum Viable:** Even worst outcomes provide ~{(p10_final / portfolio_req['required_portfolio_4pct']):.1%} of target

---

## Recommendations

### Implementation Plan
1. **Immediate:** Open investment account with low-cost ETF provider
2. **Year 1:** Invest €250,000 according to initial allocation
3. **Ongoing:** Annual rebalancing each December
4. **Age 50, 55, 60, 65:** Adjust allocation per glide path

### Risks to Monitor
1. **Market Risk:** {100-success_rate:.0f}% chance of not meeting full target
2. **Inflation Risk:** Healthcare costs may exceed 2% assumption
3. **Longevity Risk:** Living beyond 85 requires careful withdrawal management
4. **Sequence Risk:** Poor early returns significantly impact final value

### Contingency Plans
1. Consider partial annuitization at retirement (€200K → lifetime income)
2. Maintain flexibility in retirement spending (90%-110% of target)
3. Keep 3-5 years expenses in bonds approaching retirement
4. Review strategy annually and adjust if needed

---

## Appendices

### A. Detailed Analysis Files
- [Pension Projection](pension_projection.csv)
- [Gap Projection](gap_projection.csv)
- [ETF Metrics Report](etf_metrics_report.md)
- [Markowitz Optimization](markowitz_optimization.md)
- [Simulation Results](simulation_results.csv)

### B. Visualizations
- [Portfolio Simulation](../figures/glide_path_simulation.png)
- [Correlation Heatmap](../figures/correlation_heatmap.png)
- [Risk-Return Chart](../figures/risk_return_chart.png)
- [Stress Test: Market Crash](../figures/stress_market_crash.png)
- [Stress Test: Stagflation](../figures/stress_stagflation.png)

---

## Disclaimer

This analysis is for educational purposes only and does not constitute financial advice.
Past performance does not guarantee future results. Consult a qualified financial advisor
before making investment decisions.

---

**Analysis completed successfully**
"""
    
    # Save report
    report_path = os.path.join(OUTPUT['reports_dir'], 'FINAL_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Final report saved to {report_path}")
    print("\n" + "=" * 80)
    print("  ALL ANALYSES COMPLETED SUCCESSFULLY")
    print("=" * 80)


def main():
    """Run the complete analysis pipeline."""

    # Fix Windows console encoding for Unicode characters
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    start_time = time.time()

    print("\n")
    print("=" * 80)
    print(" " * 20 + "PENSION PORTFOLIO ANALYSIS")
    print(" " * 15 + "Private Retirement Planning in Germany")
    print("=" * 80)
    
    # Setup
    create_output_directories()
    
    # Load data
    returns_data = load_etf_data()
    
    try:
        # Run analysis steps
        pension_calc = run_pension_analysis()
        gap_analyzer = run_gap_analysis()
        etf_results = run_etf_analysis(returns_data)
        markowitz_results = run_markowitz_optimization(returns_data)
        simulation_results = run_glide_path_simulation(returns_data)
        run_stress_tests(returns_data)  # NOW INTEGRATED!
        
        # Generate final report
        generate_final_report(pension_calc, gap_analyzer, simulation_results)
        
        # Execution time
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\nTotal execution time: {minutes}m {seconds}s")
        print("\nAnalysis pipeline completed successfully!")
        print(f"\nAll results saved to: {OUTPUT['results_dir']}/")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
