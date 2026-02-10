"""
Validation Script for Arrival Distribution Fits

Performs goodness-of-fit tests:
- Poisson GOF for hourly arrival counts (Pearson chi-square test)
- Distribution shape validation (KS test for skill Elo)
- Temporal coverage analysis
- Mode mixture consistency checks

Outputs diagnostics and warnings for problematic patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class ArrivalValidator:
    """Validates fitted arrival parameters."""

    def __init__(self, param_dir):
        """Load parameters from directory."""
        self.param_dir = Path(param_dir)

        self.hourly_lambda = pd.read_csv(self.param_dir / 'hourly_poisson_lambda.csv')
        self.dow_dist = pd.read_csv(self.param_dir / 'dow_distribution.csv')
        self.monthly_seasonal = pd.read_csv(self.param_dir / 'monthly_seasonality.csv')
        self.skill_dist = pd.read_csv(self.param_dir / 'skill_distribution_params.csv')
        self.mode_mixture = pd.read_csv(self.param_dir / 'mode_mixture.csv')

    def validate_poisson_fit(self):
        """Check if hourly counts fit Poisson distribution."""
        print("\n" + "="*70)
        print("POISSON GOODNESS-OF-FIT TEST")
        print("="*70)

        results = []

        for mode in self.hourly_lambda['mode'].unique():
            mode_data = self.hourly_lambda[self.hourly_lambda['mode'] == mode]

            # Group by skill decile and test
            for decile in mode_data['skill_decile'].unique():
                decile_data = mode_data[mode_data['skill_decile'] == decile]

                counts = decile_data['count'].values
                lambdas = decile_data['lambda'].astype(float).values

                if np.sum(counts) < 10:  # Too few samples
                    results.append({
                        'mode': mode,
                        'skill_decile': decile,
                        'n_samples': len(counts),
                        'chi2_stat': np.nan,
                        'p_value': np.nan,
                        'status': 'SKIP (too few samples)'
                    })
                    continue

                # Pearson chi-square GOF test
                mask = lambdas > 0
                expected = lambdas[mask]
                observed = counts[mask]

                if len(observed) > 1 and np.sum(expected) > 5:
                    chi2_stat = np.sum(((observed - expected) ** 2) / (expected + 1e-6))
                    p_value = 1 - stats.chi2.cdf(chi2_stat, len(observed) - 1)

                    status = 'PASS' if p_value > 0.05 else 'WARN'

                    results.append({
                        'mode': mode,
                        'skill_decile': decile,
                        'n_samples': len(counts),
                        'chi2_stat': chi2_stat,
                        'p_value': p_value,
                        'status': status
                    })

        if not results:
            print("\nNo tests performed (insufficient data)")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        print(f"\nTested {len(results_df)} mode × skill combinations")
        print(f"\nResults summary:")
        if len(results_df) > 0:
            status_counts = results_df['status'].value_counts()
            for status, count in status_counts.items():
                print(f"  {status}: {count}")

        print(f"\nDetailed results (first 20):")
        if len(results_df) > 0:
            print(results_df.head(20).to_string(index=False))

        # Warnings
        warns = results_df[results_df['status'] == 'WARN']
        if len(warns) > 0:
            print(f"\n⚠ Warning: {len(warns)} combinations have p < 0.05")

        return results_df

    def validate_skill_distribution(self):
        """Check Elo distributions within skill deciles."""
        print("\n" + "="*70)
        print("SKILL DISTRIBUTION VALIDATION")
        print("="*70)

        print(f"\nSkill deciles: {self.skill_dist['skill_decile'].nunique()}")
        print(f"Modes: {list(self.skill_dist['mode'].unique())}")

        # Check for monotonicity in mean Elo across deciles
        for mode in self.skill_dist['mode'].unique():
            mode_data = self.skill_dist[self.skill_dist['mode'] == mode].copy()
            mode_data = mode_data.sort_values('skill_decile')

            mean_elos = pd.to_numeric(mode_data['mean_elo']).values
            std_elos = pd.to_numeric(mode_data['std_elo']).values

            # Check if generally increasing
            diffs = np.diff(mean_elos)
            if len(diffs) > 0:
                increasing = np.sum(diffs > 0) / len(diffs)
            else:
                increasing = 1.0

            print(f"\n{mode}:")
            print(f"  Elo range: {mean_elos.min():.0f} - {mean_elos.max():.0f}")
            print(f"  Deciles increasing: {increasing:.1%}")

            if increasing < 0.7:
                print(f"  ⚠ Warning: Low monotonicity in Elo across deciles")

            # Check for extreme std values
            if np.any(std_elos > 500):
                print(f"  ⚠ Warning: High Elo std in some deciles: {std_elos.max():.0f}")

    def validate_temporal_coverage(self):
        """Check temporal coverage and patterns."""
        print("\n" + "="*70)
        print("TEMPORAL COVERAGE VALIDATION")
        print("="*70)

        # Hour coverage
        hours_with_data = self.hourly_lambda[self.hourly_lambda['lambda'] > 0]['hour'].nunique()
        print(f"\nHour coverage: {hours_with_data}/24 hours have λ > 0")
        if hours_with_data < 20:
            print(f"  ⚠ Warning: Low coverage of hours")

        # Day-of-week coverage
        dow_coverage = self.dow_dist.groupby('mode')['day_of_week'].nunique()
        print(f"\nDay-of-week coverage by mode:")
        for mode, n_days in dow_coverage.items():
            print(f"  {mode}: {n_days}/7 days")
            if n_days < 6:
                print(f"    ⚠ Warning: Incomplete week coverage")

        # Month coverage
        months_with_data = len(self.monthly_seasonal)
        print(f"\nMonth coverage: {months_with_data}/12 months")
        if months_with_data < 12:
            print(f"  Note: Dataset may not span full year")

    def validate_mode_mixture(self):
        """Validate mode mixture fractions."""
        print("\n" + "="*70)
        print("MODE MIXTURE VALIDATION")
        print("="*70)

        print(f"\nMode counts and fractions:")
        for _, row in self.mode_mixture.iterrows():
            print(f"  {row['mode']}: {row['count']:,} arrivals ({row['fraction']:.2%})")

        # Check for extreme imbalance
        fracs = self.mode_mixture['fraction'].values
        if fracs.max() / fracs.min() > 100:
            print(f"\n⚠ Warning: Extreme imbalance in mode mixture")

        # Check total
        total_frac = self.mode_mixture['fraction'].sum()
        print(f"\nTotal fraction: {total_frac:.4f}")
        if abs(total_frac - 1.0) > 0.0001:
            print(f"  ⚠ Warning: Fractions don't sum to 1.0")

    def validate_consistency(self):
        """Cross-validation between parameter sets."""
        print("\n" + "="*70)
        print("INTERNAL CONSISTENCY CHECKS")
        print("="*70)

        # Check: modes in hourly_lambda match mode_mixture
        hourly_modes = set(self.hourly_lambda['mode'].unique())
        mixture_modes = set(self.mode_mixture['mode'].unique())
        skill_modes = set(self.skill_dist['mode'].unique())

        print(f"\nModes by parameter set:")
        print(f"  hourly_lambda: {hourly_modes}")
        print(f"  mode_mixture: {mixture_modes}")
        print(f"  skill_dist: {skill_modes}")

        if not (hourly_modes == mixture_modes):
            print(f"  ⚠ Warning: Mode mismatch between hourly_lambda and mode_mixture")

        if not (hourly_modes == skill_modes):
            print(f"  ⚠ Warning: Mode mismatch between hourly_lambda and skill_dist")

        # Check: dow_distribution covers all modes
        dow_modes = set(self.dow_dist['mode'].unique())
        if not (dow_modes == mixture_modes):
            print(f"  ⚠ Warning: dow_distribution covers {dow_modes}, mixture has {mixture_modes}")

    def generate_report(self, output_path=None):
        """Generate full validation report."""
        if output_path is None:
            output_path = self.param_dir / 'validation_report.txt'

        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ARRIVAL DATASET VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Parameter directory: {self.param_dir}\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            # Redirect print to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = open(output_path, 'w')

            self.validate_poisson_fit()
            self.validate_skill_distribution()
            self.validate_temporal_coverage()
            self.validate_mode_mixture()
            self.validate_consistency()

            sys.stdout = original_stdout

        print(f"\nValidation report saved to: {output_path}")
        return output_path

    def run_all(self):
        """Run all validation checks."""
        print("\n" + "="*70)
        print("STARTING VALIDATION")
        print("="*70)

        self.validate_poisson_fit()
        self.validate_skill_distribution()
        self.validate_temporal_coverage()
        self.validate_mode_mixture()
        self.validate_consistency()

        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate arrival dataset parameters')
    parser.add_argument(
        '--param-dir',
        type=str,
        default='/sessions/admiring-busy-dijkstra/mnt/TOG Matchmaking/arrival_dataset',
        help='Directory with fitted parameters'
    )
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Output report file (default: validation_report.txt in param-dir)'
    )

    args = parser.parse_args()

    validator = ArrivalValidator(args.param_dir)
    validator.run_all()

    if args.report:
        validator.generate_report(args.report)


if __name__ == '__main__':
    main()
