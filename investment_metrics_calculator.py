"""
Investment metrics calculator (minimal edits for config consistency)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from config_file import DATA_FILES, SIMULATION
from plotting_config import COLOR_SCHEME, PLOT_SIZES, apply_common_formatting

class InvestmentMetricsCalculator:

    def __init__(self, returns_data: pd.DataFrame = None, csv_file_path: str = None):
        """Initialize with DataFrame (decimal returns) or CSV file path."""
        if returns_data is not None:
            self.returns_data = returns_data
            self.csv_file_path = None
            print(f"Data loaded: {len(self.returns_data.columns)} instruments, {len(self.returns_data)} observations")
            if len(self.returns_data) > 0:
                print(f"Date range: {self.returns_data.index.min()} to {self.returns_data.index.max()}")
        else:
            if csv_file_path is None:
                csv_file_path = DATA_FILES['etf_returns']
            self.csv_file_path = csv_file_path
            self.returns_data = None
            self._load_data()

    def _load_data(self):
        try:
            data = pd.read_csv(self.csv_file_path, encoding='utf-8-sig')

            try:
                data['Dates'] = pd.to_datetime(data['Dates'], format=DATA_FILES.get('date_format'), dayfirst=True)
            except Exception:
                data['Dates'] = pd.to_datetime(data['Dates'], dayfirst=True, errors='coerce')

            data.set_index('Dates', inplace=True)
            data = data.replace('#N/A N/A', np.nan)
            data = data.apply(pd.to_numeric, errors='coerce')
            data = data.dropna(axis=1, how='all')

            self.returns_data = data / 100  # Convert to decimals

            print(f"Data loaded: {len(self.returns_data.columns)} instruments, {len(self.returns_data)} observations")
            print(f"Date range: {self.returns_data.index.min()} to {self.returns_data.index.max()}")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def calculate_average_returns(self, annualized=True):
        avg_returns = self.returns_data.mean(skipna=True)
        if annualized:
            avg_returns = avg_returns * SIMULATION['trading_days_per_year']
        return avg_returns

    def calculate_volatility(self, annualized=True):
        volatility = self.returns_data.std(skipna=True)
        if annualized:
            volatility = volatility * np.sqrt(SIMULATION['trading_days_per_year'])
        return volatility

    def calculate_correlation_matrix(self):
        return self.returns_data.corr()

    def calculate_covariance_matrix(self, annualized=True):
        cov_matrix = self.returns_data.cov()
        if annualized:
            cov_matrix = cov_matrix * SIMULATION['trading_days_per_year']
        return cov_matrix

    def calculate_var(self, confidence_level=0.05):
        var_results = {}
        for col in self.returns_data.columns:
            valid_data = self.returns_data[col].dropna()
            if len(valid_data) > 0:
                var_results[col] = valid_data.quantile(confidence_level)
            else:
                var_results[col] = np.nan
        return pd.Series(var_results)

    def calculate_sharpe_ratio(self, risk_free_rate=0.0277):
        avg_returns = self.calculate_average_returns(annualized=True)
        volatility = self.calculate_volatility(annualized=True)
        return (avg_returns - risk_free_rate) / volatility

    def generate_summary_report(self):
        avg_returns = self.calculate_average_returns(annualized=True)
        volatility = self.calculate_volatility(annualized=True)
        sharpe_ratio = self.calculate_sharpe_ratio()
        var_95 = self.calculate_var(confidence_level=0.05)
        var_99 = self.calculate_var(confidence_level=0.01)
        correlation_matrix = self.calculate_correlation_matrix()
        covariance_matrix = self.calculate_covariance_matrix(annualized=True)

        summary_df = pd.DataFrame({
            'Annual Return (%)': avg_returns * 100,
            'Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'VaR 95% (Daily %)': var_95 * 100,
            'VaR 99% (Daily %)': var_99 * 100
        })

        summary_df = summary_df.sort_values('Annual Return (%)', ascending=False)

        results = {
            'summary_statistics': summary_df,
            'correlation_matrix': correlation_matrix,
            'covariance_matrix': covariance_matrix,
            'data_info': {
                'start_date': self.returns_data.index.min(),
                'end_date': self.returns_data.index.max(),
                'num_observations': len(self.returns_data),
                'num_instruments': len(self.returns_data.columns)
            }
        }

        return results

    def plot_correlation_heatmap(self, save_path='correlation_heatmap.png'):
        correlation_matrix = self.calculate_correlation_matrix()
        fig, ax = plt.subplots(figsize=PLOT_SIZES['square'])
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f', square=True, ax=ax)
        apply_common_formatting(ax, title='Asset Correlation Matrix')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Correlation heatmap saved to {save_path}")

    def plot_returns_and_volatility(self, save_path='risk_return_chart.png'):
        avg_returns = self.calculate_average_returns(annualized=True) * 100
        volatility = self.calculate_volatility(annualized=True) * 100

        fig, ax = plt.subplots(figsize=PLOT_SIZES['large'])
        ax.scatter(volatility, avg_returns, s=100, alpha=0.7, color=COLOR_SCHEME['base'])

        for instrument in avg_returns.index:
            ax.annotate(instrument, (volatility[instrument], avg_returns[instrument]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9, ha='left')

        apply_common_formatting(ax, title='Risk-Return Profile',
                              xlabel='Annual Volatility (%)', ylabel='Average Annual Return (%)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Risk-return chart saved to {save_path}")

    def save_results_to_markdown(self, filename='investment_metrics_report.md'):
        results = self.generate_summary_report()

        data_availability = []
        for col in results['summary_statistics'].index:
            valid_data = self.returns_data[col].dropna()
            if len(valid_data) > 0:
                first_date = valid_data.index.min()
                last_date = valid_data.index.max()
                obs_count = len(valid_data)
                data_availability.append({
                    'Asset': col,
                    'Start Date': first_date.strftime('%Y'),
                    'End Date': last_date.strftime('%Y'),
                    'Observations': f"{obs_count:,}"
                })

        data_avail_df = pd.DataFrame(data_availability)

        markdown_content = f"""# Investment Metrics Report

Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}

## Summary

| Metric | Best Asset | Value |
|--------|------------|-------|
| Highest Return | {results['summary_statistics'].index[0]} | {results['summary_statistics'].iloc[0, 0]:.2f}% |
| Best Risk-Adjusted | {results['summary_statistics'].sort_values('Sharpe Ratio', ascending=False).index[0]} | {results['summary_statistics']['Sharpe Ratio'].max():.3f} Sharpe |
| Lowest Risk | {results['summary_statistics'].sort_values('Volatility (%)', ascending=True).index[0]} | {results['summary_statistics']['Volatility (%)'].min():.2f}% Vol |

## Data Overview

**Analysis Period**: {results['data_info']['start_date'].strftime('%B %Y')} to {results['data_info']['end_date'].strftime('%B %Y')}
**Total Investment Vehicles**: {results['data_info']['num_instruments']}
**Maximum Observations**: {results['data_info']['num_observations']:,} daily returns

### Data Availability

{data_avail_df.to_markdown(index=False)}

## Performance Metrics

{results['summary_statistics'].round(2).to_markdown()}

## Correlation Matrix

{results['correlation_matrix'].round(2).to_markdown()}

## Covariance Matrix (Annualized)

{results['covariance_matrix'].round(3).to_markdown()}

## Charts

![Risk-Return Profile](risk_return_chart.png)
![Correlation Heatmap](correlation_heatmap.png)

## Investment Recommendations

| Investor Profile | Recommended Assets | Rationale |
|------------------|-------------------|-----------|
| Growth Seeker | {results['summary_statistics'].index[0]} | Highest annual return ({results['summary_statistics'].iloc[0, 0]:.1f}%) |
| Balanced Investor | {results['summary_statistics'].sort_values('Sharpe Ratio', ascending=False).index[0]} | Best risk-adjusted performance (Sharpe: {results['summary_statistics']['Sharpe Ratio'].max():.2f}) |
| Conservative | {results['summary_statistics'].sort_values('Volatility (%)', ascending=True).index[0]} | Lowest volatility ({results['summary_statistics']['Volatility (%)'].min():.1f}%) |

"""

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"Report saved to {filename}")


def main():
    print("Investment Metrics Analysis")
    print("-" * 40)

    calculator = InvestmentMetricsCalculator()

    if calculator.data is None:
        print("Failed to load data.")
        return

    print("Calculating metrics...")
    results = calculator.generate_summary_report()

    print("Creating visualizations...")
    calculator.plot_correlation_heatmap(
        save_path='results/figures/correlation_heatmap.png'
    )
    calculator.plot_returns_and_volatility(
        save_path='results/figures/risk_return_chart.png'
    )

    print("Generating report...")
    calculator.save_results_to_markdown(filename='results/reports/investment_metrics_report.md')

    summary_stats = results['summary_statistics']
    print(f"\nBest Performer: {summary_stats.index[0]} ({summary_stats.iloc[0, 0]:.2f}%)")
    print(f"Best Risk-Adjusted: {summary_stats.sort_values('Sharpe Ratio', ascending=False).index[0]} (Sharpe: {summary_stats['Sharpe Ratio'].max():.3f})")
    print(f"Lowest Risk: {summary_stats.sort_values('Volatility (%)', ascending=True).index[0]} ({summary_stats['Volatility (%)'].min():.2f}% vol)")

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
