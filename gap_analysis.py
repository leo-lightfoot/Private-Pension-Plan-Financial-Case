"""
Retirement Income Gap Analysis (updated)

Key changes:
- Compute birth year from CLIENT['starting_year'] and CLIENT['current_age'].
- Compute retirement_year as birth_year + CLIENT['retirement_age'] (consistent calendar).
- Use GermanPensionCalculator.calculate_salary_for_year to get the final gross salary.
- Keep PKV step logic consistent with config (uses PKV['increase_frequency_years']).
"""

import numpy as np
import pandas as pd
from config_file import GAP, PKV, SALARY, CLIENT, TAX_RATES
from pension_calculator import GermanPensionCalculator


class RetirementGapAnalyzer:
    """
    Analyze the gap between retirement income needs and statutory pension.
    """

    def __init__(self):
        """Initialize with parameters from config and pension calculator."""
        self.pension_calculator = GermanPensionCalculator()
        self.replacement_rate = GAP['replacement_rate']
        self.inflation_rate = GAP['inflation_rate']
        self.withdrawal_rate = GAP['withdrawal_rate']

        # Derive calendar years consistently
        # birth_year = starting_year - current_age
        self.birth_year = CLIENT['starting_year'] - CLIENT['current_age']
        # retirement year is birth_year + retirement_age
        self.retirement_year = self.birth_year + CLIENT['retirement_age']

    def calculate_final_net_salary(self):
        """
        Calculate net salary in the final working year (the year BEFORE retirement).
        Uses GermanPensionCalculator.calculate_salary_for_year for gross salary,
        then deducts social security, income tax and PKV.

        Returns:
            dict: Components and final net salary
        """
        final_working_year = self.retirement_year - 1

        # Gross salary at final working year
        final_gross = self.pension_calculator.calculate_salary_for_year(final_working_year)

        # Social security deductions (employee share): pension + unemployment + care
        # Use values from TAX_RATES config for consistency
        social_security_rate = (
            TAX_RATES['pension_contribution'] +
            TAX_RATES['unemployment_insurance'] +
            TAX_RATES['care_insurance']
        )
        social_security = final_gross * social_security_rate

        # Income tax (simplified effective)
        taxable_income = final_gross - social_security
        income_tax = taxable_income * TAX_RATES['income_tax_rate']

        # PKV (private health insurance) in final working year: discrete steps every N years
        years_from_2025 = final_working_year - 2025
        freq = PKV.get('increase_frequency_years', 3)
        steps = max(0, years_from_2025 // freq)
        pkv_monthly = PKV['monthly_cost_age_45'] * (1 + PKV['increase_rate']) ** steps
        pkv_annual = pkv_monthly * 12

        # Net salary
        net_salary = final_gross - social_security - income_tax - pkv_annual

        return {
            'gross_salary': final_gross,
            'social_security': social_security,
            'income_tax': income_tax,
            'pkv_annual': pkv_annual,
            'net_salary': net_salary,
            'final_working_year': final_working_year
        }

    def calculate_retirement_needs(self):
        """
        Calculate annual income needed in retirement:
            Needs = Replacement Rate × Final Net Salary + PKV (in retirement)
        """
        final_salary_details = self.calculate_final_net_salary()
        final_net = final_salary_details['net_salary']

        # Income replacement need (90% of final net salary per config)
        income_replacement = final_net * self.replacement_rate

        # PKV in retirement (first retirement year)
        years_from_2025 = self.retirement_year - 2025
        freq = PKV.get('increase_frequency_years', 3)
        steps = max(0, years_from_2025 // freq)
        pkv_monthly_retirement = PKV['monthly_cost_age_45'] * (1 + PKV['increase_rate']) ** steps
        pkv_annual_retirement = pkv_monthly_retirement * 12

        total_needs = income_replacement + pkv_annual_retirement

        return {
            'final_net_salary': final_net,
            'income_replacement_90pct': income_replacement,
            'pkv_retirement': pkv_annual_retirement,
            'total_annual_needs': total_needs
        }

    def calculate_pension_gap(self):
        """
        Calculate annual gap = Total Needs - Net Statutory Pension
        """
        needs = self.calculate_retirement_needs()

        # Net statutory pension using pension_calculator for retirement_year
        pension_details = self.pension_calculator.calculate_net_pension(self.retirement_year)
        net_pension = pension_details['net_pension']

        annual_gap = needs['total_annual_needs'] - net_pension

        return {
            'retirement_needs': needs['total_annual_needs'],
            'net_statutory_pension': net_pension,
            'annual_gap': annual_gap,
            'monthly_gap': annual_gap / 12,
            'breakdown': {
                'needs': needs,
                'pension': pension_details
            }
        }

    def calculate_required_portfolio_value(self):
        """
        Required portfolio value at retirement to close the gap using 4% withdrawal rule
        """
        gap_analysis = self.calculate_pension_gap()
        annual_gap = gap_analysis['annual_gap']

        required_value_4pct = annual_gap / self.withdrawal_rate
        required_value_3pct = annual_gap / 0.03
        required_value_3_5pct = annual_gap / 0.035

        return {
            'annual_gap': annual_gap,
            'required_portfolio_4pct': required_value_4pct,
            'required_portfolio_3_5pct': required_value_3_5pct,
            'required_portfolio_3pct': required_value_3pct,
            'initial_investment': CLIENT['initial_investment'],
            'required_growth_4pct': required_value_4pct / CLIENT['initial_investment'],
            'required_cagr_4pct': (required_value_4pct / CLIENT['initial_investment']) ** (1 / (self.retirement_year - CLIENT['starting_year'])) - 1
        }

    def project_gap_with_inflation(self, years_in_retirement=20):
        """
        Project the gap over retirement years with inflation.
        """
        base_gap = self.calculate_pension_gap()['annual_gap']

        data = []
        for year_idx in range(years_in_retirement):
            year = self.retirement_year + year_idx
            age = CLIENT['retirement_age'] + year_idx
            inflated_gap = base_gap * (1 + self.inflation_rate) ** year_idx
            cumulative = sum([base_gap * (1 + self.inflation_rate) ** i for i in range(year_idx + 1)])
            data.append({
                'Year': year,
                'Age': age,
                'Annual Gap (EUR)': inflated_gap,
                'Monthly Gap (EUR)': inflated_gap / 12,
                'Cumulative Gap (EUR)': cumulative
            })

        return pd.DataFrame(data)

    def generate_longevity_analysis(self):
        """
        Returns cumulative gap totals for plausible longevity scenarios.
        """
        scenarios = {
            'Conservative (to age 80)': 12,  # 12 years in retirement
            'Median (to age 85)': 17,
            'Longevity (to age 90)': 22,
            'Extended (to age 95)': 27,
        }

        base_gap = self.calculate_pension_gap()['annual_gap']
        results = {}
        for name, years in scenarios.items():
            cumulative = sum([base_gap * (1 + self.inflation_rate) ** i for i in range(years)])
            results[name] = cumulative

        return results

    def print_summary(self):
        """Print human-readable summary."""
        print("=" * 70)
        print("RETIREMENT INCOME GAP ANALYSIS")
        print("=" * 70)

        # Final salary and deductions
        salary_details = self.calculate_final_net_salary()
        print("\n💰 Final Working Year (Age {}, Year {}):".format(CLIENT['retirement_age'] - 1, salary_details['final_working_year']))
        print(f"   Gross salary: €{salary_details['gross_salary']:,.0f}")
        print(f"   - Social security: €{salary_details['social_security']:,.0f}")
        print(f"   - Income tax: €{salary_details['income_tax']:,.0f}")
        print(f"   - PKV: €{salary_details['pkv_annual']:,.0f}")
        print(f"   Net salary: €{salary_details['net_salary']:,.0f}")

        # Needs and pension
        needs = self.calculate_retirement_needs()
        print("\n🎯 Retirement Income Needs (Age {}, Year {}):".format(CLIENT['retirement_age'], self.retirement_year))
        print(f"   90% of final net salary: €{needs['income_replacement_90pct']:,.0f}")
        print(f"   + PKV in retirement: €{needs['pkv_retirement']:,.0f}")
        print(f"   Total annual needs: €{needs['total_annual_needs']:,.0f}")

        gap = self.calculate_pension_gap()
        pension = gap['breakdown']['pension']
        print("\n🏛️  German Statutory Pension:")
        print(f"   Gross pension: €{pension['gross_pension']:,.0f}")
        print(f"   - Pension tax: €{pension['pension_tax']:,.0f}")
        print(f"   - PKV: €{pension['pkv_annual']:,.0f}")
        print(f"   Net pension: €{gap['net_statutory_pension']:,.0f}")

        print("\n⚠️  RETIREMENT GAP:")
        print(f"   Annual gap: €{gap['annual_gap']:,.0f}")
        print(f"   Monthly gap: €{gap['monthly_gap']:,.0f}")

        req = self.calculate_required_portfolio_value()
        print("\n📊 Required Portfolio Value at Retirement:")
        print(f"   Using 4.0% withdrawal rate: €{req['required_portfolio_4pct']:,.0f}")
        print(f"   Required CAGR (to reach target): {req['required_cagr_4pct']:.2%}")

        print("\n" + "=" * 70)


def main():
    analyzer = RetirementGapAnalyzer()
    analyzer.print_summary()
    print("\nLongevity analysis (summary):")
    print(analyzer.generate_longevity_analysis())
    print("\nGap projection (first 5 years):")
    print(analyzer.project_gap_with_inflation(years_in_retirement=5).head().to_string(index=False))


if __name__ == "__main__":
    main()
