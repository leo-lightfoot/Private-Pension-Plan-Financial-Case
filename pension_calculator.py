"""
German Statutory Pension Calculator (updated salary profile)
- Uses case-provided salary_by_age table and linear interpolation between anchors.
- After age 45 salary grows only with inflation (SALARY['inflation_rate']).
"""

import numpy as np
import pandas as pd
from config_file import SALARY, GERMAN_PENSION, CLIENT, PKV

class GermanPensionCalculator:
    """
    Calculate German statutory pension based on salary history and pension points.
    """

    def __init__(self):
        """Initialize with parameters from config."""
        # Age at career start (case)
        self.start_age = 30  # started at age 30 in SALARY['age_30_year']
        self.start_year = SALARY.get('age_30_year', 2010)  # 2010 per case
        self.current_age = CLIENT['current_age']
        self.retirement_age = CLIENT['retirement_age']

        # Salary structure (case-driven)
        self.salary_by_age = SALARY['salary_by_age']
        self.inflation_rate = SALARY['inflation_rate']

        # Pension parameters
        self.durchschnittsentgelt_2025 = GERMAN_PENSION['durchschnittsentgelt_2025']
        self.rentenwert_2025 = GERMAN_PENSION['aktueller_rentenwert_2025']
        self.pension_inflation = GERMAN_PENSION['inflation_rate']

    def _age_from_year(self, year):
        """Return age corresponding to calendar year."""
        return self.start_age + (year - self.start_year)

    def calculate_salary_for_year(self, year):
        """
        Calculate gross salary for a given calendar year using:
          - exact table values at anchor ages (30,35,40,45)
          - linear interpolation between anchor ages
          - inflation-only growth after age 45

        Args:
            year (int): Calendar year

        Returns:
            float: Gross annual salary in EUR
        """
        years_since_start = year - self.start_year
        if years_since_start < 0:
            return 0.0

        age = self._age_from_year(year)

        # If exact anchor age available, return it
        if age in self.salary_by_age:
            return float(self.salary_by_age[age])

        # Sort anchor ages
        anchors = sorted(self.salary_by_age.keys())

        # If age is between two anchors, do linear interpolation
        for a0, a1 in zip(anchors[:-1], anchors[1:]):
            if a0 < age < a1:
                s0 = self.salary_by_age[a0]
                s1 = self.salary_by_age[a1]
                weight = (age - a0) / (a1 - a0)
                return float(s0 * (1 - weight) + s1 * weight)

        # If age is below first anchor (shouldn't happen since start_age == first anchor), fallback
        if age < anchors[0]:
            return float(self.salary_by_age[anchors[0]])

        # If age is above last anchor ( >45 ), grow with inflation from age 45 salary
        last_anchor_age = anchors[-1]
        last_salary = float(self.salary_by_age[last_anchor_age])
        years_after_anchor = age - last_anchor_age
        return float(last_salary * (1 + self.inflation_rate) ** years_after_anchor)

    def calculate_durchschnittsentgelt(self, year):
        """
        Calculate German average earnings (Durchschnittsentgelt) for a given year.
        Grows with pension inflation from 2025 baseline.
        """
        if year < 2025:
            years_before_2025 = 2025 - year
            return self.durchschnittsentgelt_2025 / (1 + self.pension_inflation) ** years_before_2025
        else:
            years_after_2025 = year - 2025
            return self.durchschnittsentgelt_2025 * (1 + self.pension_inflation) ** years_after_2025

    def calculate_pension_points(self, year):
        """
        Pension Points = Individual Salary ÷ Average Earnings (for that year)
        """
        salary = self.calculate_salary_for_year(year)
        avg_earnings = self.calculate_durchschnittsentgelt(year)

        if salary <= 0 or avg_earnings <= 0:
            return 0.0

        return salary / avg_earnings

    def calculate_total_pension_points(self, retirement_year=None):
        """
        Sum pension points year-by-year from career start up to retirement (exclusive of retirement year).
        """
        if retirement_year is None:
            retirement_year = self.start_year + (self.retirement_age - self.start_age)

        total_points = 0.0
        for year in range(self.start_year, retirement_year):
            total_points += self.calculate_pension_points(year)
        return total_points

    def calculate_rentenwert(self, year):
        """
        Project pension value (Rentenwert) from 2025 baseline using pension_inflation.
        """
        if year < 2025:
            years_before_2025 = 2025 - year
            return self.rentenwert_2025 / (1 + self.pension_inflation) ** years_before_2025
        else:
            years_after_2025 = year - 2025
            return self.rentenwert_2025 * (1 + self.pension_inflation) ** years_after_2025

    def calculate_gross_pension(self, retirement_year=None):
        """
        Annual Pension = Total Points × Rentenwert (at retirement year)
        """
        if retirement_year is None:
            retirement_year = self.start_year + (self.retirement_age - self.start_age)

        total_points = self.calculate_total_pension_points(retirement_year)
        rentenwert = self.calculate_rentenwert(retirement_year)
        return total_points * rentenwert

    def calculate_net_pension(self, retirement_year=None):
        """
        Simplified net pension after pension tax and PKV contributions.
        Net = Gross - pension_tax - PKV_annual
        """
        if retirement_year is None:
            retirement_year = self.start_year + (self.retirement_age - self.start_age)

        gross_pension = self.calculate_gross_pension(retirement_year)

        pension_tax = gross_pension * GERMAN_PENSION['pension_tax_rate']

        # PKV in retirement: discrete step increases every increase_frequency_years
        years_to_retirement = retirement_year - 2025
        # compute number of steps since 2025 to retirement year based on PKV settings in config
        freq = PKV.get('increase_frequency_years', 3)
        steps = max(0, years_to_retirement // freq)
        pkv_monthly = PKV['monthly_cost_age_45'] * (1 + PKV['increase_rate']) ** steps
        pkv_annual = pkv_monthly * 12

        net_pension = gross_pension - pension_tax - pkv_annual

        return {
            'gross_pension': gross_pension,
            'pension_tax': pension_tax,
            'pkv_annual': pkv_annual,
            'net_pension': net_pension,
            'total_deductions': pension_tax + pkv_annual
        }

    def generate_career_summary(self):
        """
        Year-by-year table: Year, Age, Gross Salary, Average Earnings, Pension Points, Cumulative Points
        """
        retirement_year = self.start_year + (self.retirement_age - self.start_age)
        data = []
        cumulative_points = 0.0
        for year in range(self.start_year, retirement_year):
            age = self._age_from_year(year)
            salary = self.calculate_salary_for_year(year)
            avg_earnings = self.calculate_durchschnittsentgelt(year)
            points = self.calculate_pension_points(year)
            cumulative_points += points
            data.append({
                'Year': year,
                'Age': age,
                'Gross Salary': salary,
                'Average Earnings': avg_earnings,
                'Pension Points': points,
                'Cumulative Points': cumulative_points
            })
        df = pd.DataFrame(data)
        return df

    def print_summary(self):
        """Human-readable summary (keeps previous formatting)."""
        retirement_year = self.start_year + (self.retirement_age - self.start_age)

        print("=" * 70)
        print("GERMAN STATUTORY PENSION CALCULATION SUMMARY")
        print("=" * 70)
        print(f"\n📋 Career Overview:")
        print(f"   Started working: Age {self.start_age} (Year {self.start_year})")
        print(f"   Current age: {self.current_age} (Year {CLIENT['starting_year']})")
        print(f"   Retirement age: {self.retirement_age} (Year {retirement_year})")
        print(f"   Total career years: {self.retirement_age - self.start_age}")

        print(f"\n💰 Salary Information:")
        print(f"   Salary anchors (age: salary): {self.salary_by_age}")
        print(f"   Inflation rate: {self.inflation_rate:.1%}")
        current_salary = self.calculate_salary_for_year(CLIENT['starting_year'])
        print(f"   Current salary (age {self.current_age}): €{current_salary:,.0f}")
        retirement_salary = self.calculate_salary_for_year(retirement_year - 1)
        print(f"   Final salary (age {self.retirement_age - 1}): €{retirement_salary:,.0f}")

        total_points = self.calculate_total_pension_points(retirement_year)
        print(f"\n🎯 Pension Points:")
        print(f"   Total accumulated points: {total_points:.2f}")
        avg_points = total_points / (self.retirement_age - self.start_age)
        print(f"   Average points per year: {avg_points:.2f}")

        print(f"\n📊 Pension Calculation:")
        print(f"   Pension value (at retirement): €{self.calculate_rentenwert(retirement_year):,.2f} per point")

        pension_details = self.calculate_net_pension(retirement_year)
        print(f"   Gross annual pension: €{pension_details['gross_pension']:,.0f}")
        print(f"   - Pension tax ({GERMAN_PENSION['pension_tax_rate']*100:.1f}%): €{pension_details['pension_tax']:,.0f}")
        print(f"   - PKV (health insurance): €{pension_details['pkv_annual']:,.0f}")
        print(f"   Net annual pension: €{pension_details['net_pension']:,.0f}")
        print(f"   Monthly net pension: €{pension_details['net_pension']/12:,.0f}")

        print("\n" + "=" * 70)


def main():
    calculator = GermanPensionCalculator()
    calculator.print_summary()
    career_df = calculator.generate_career_summary()
    print("\nFirst rows of career summary:")
    print(career_df.head().to_string(index=False))
    # Save to CSV if desired:
    # career_df.to_csv('results/pension_projection.csv', index=False)


if __name__ == "__main__":
    main()
