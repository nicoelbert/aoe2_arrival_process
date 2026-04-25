"""
Synthetic Arrival Generator for AoE2 Matchmaking

Generates synthetic arrival events from fitted parameters:
- Samples mode from mode_mixture distribution
- Samples skill decile from skill distribution
- Samples hour and day-of-week from temporal distributions
- Samples arrival count from Poisson distribution with rate λ
- Applies monthly seasonality multiplier

Output: synthetic arrival times with mode and skill level
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse


class ArrivalGenerator:
    """Generates synthetic matchmaking arrivals from fitted parameters."""

    def __init__(self, param_dir):
        """Load fitted parameters from directory."""
        self.param_dir = Path(param_dir)

        print("Loading fitted parameters...")
        self.hourly_lambda = pd.read_csv(self.param_dir / 'hourly_poisson_lambda.csv')
        self.dow_dist = pd.read_csv(self.param_dir / 'dow_distribution.csv')
        self.monthly_seasonal = pd.read_csv(self.param_dir / 'monthly_seasonality.csv')
        self.skill_dist = pd.read_csv(self.param_dir / 'skill_distribution_params.csv')
        self.mode_mixture = pd.read_csv(self.param_dir / 'mode_mixture.csv')

        print(f"  Modes: {', '.join(self.mode_mixture['mode'].values)}")
        print(f"  Skill deciles: 1-10")

    def sample_mode(self, rng):
        """Sample mode from mixture distribution."""
        return rng.choice(self.mode_mixture['mode'].values,
                         p=self.mode_mixture['fraction'].values)

    def sample_skill_decile(self, mode, rng):
        """Sample skill decile for a given mode."""
        mode_skills = self.skill_dist[self.skill_dist['mode'] == mode]
        if len(mode_skills) == 0:
            return rng.randint(1, 11)
        return rng.choice(mode_skills['skill_decile'].values)

    def sample_hour_dow(self, mode, skill_decile, rng):
        """Sample (hour, day_of_week) pair."""
        # Get hourly lambda values for this mode and skill
        mode_skill_data = self.hourly_lambda[
            (self.hourly_lambda['mode'] == mode) &
            (self.hourly_lambda['skill_decile'] == skill_decile)
        ]

        if len(mode_skill_data) == 0:
            hour = rng.randint(0, 24)
            dow = rng.randint(0, 7)
            return hour, dow

        # Sample hour weighted by lambda values
        hours = mode_skill_data['hour'].values
        lambdas = mode_skill_data['lambda'].values
        if lambdas.sum() > 0:
            hour = rng.choice(hours, p=lambdas / lambdas.sum())
        else:
            hour = rng.choice(hours)

        # Sample dow weighted by dow distribution for this mode
        mode_dow = self.dow_dist[self.dow_dist['mode'] == mode]
        dows = mode_dow['day_of_week'].values
        fracs = mode_dow['fraction'].values
        dow = rng.choice(dows, p=fracs / fracs.sum())

        return int(hour), int(dow)

    def sample_elo(self, mode, skill_decile, rng):
        """Sample Elo from skill decile distribution."""
        decile_data = self.skill_dist[
            (self.skill_dist['mode'] == mode) &
            (self.skill_dist['skill_decile'] == skill_decile)
        ]

        if len(decile_data) == 0:
            return 1600.0

        row = decile_data.iloc[0]
        mean_elo = row['mean_elo']
        std_elo = row['std_elo']

        # Sample from truncated normal (within decile bounds)
        min_elo = row['min_elo']
        max_elo = row['max_elo']

        elo = rng.normal(mean_elo, max(std_elo, 1.0))
        elo = np.clip(elo, min_elo, max_elo)

        return float(elo)

    def generate(self, n_days=30, start_date=None, rng_seed=None):
        """
        Generate synthetic arrivals.

        Parameters:
        -----------
        n_days : int
            Number of days to simulate
        start_date : datetime or str
            Starting date (default: today)
        rng_seed : int
            Random seed for reproducibility

        Returns:
        --------
        arrivals : DataFrame
            Columns: timestamp, hour, day_of_week, day_name, mode, skill_decile,
            elo, month, month_name
        """
        rng = np.random.RandomState(rng_seed)

        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)

        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                     'Friday', 'Saturday', 'Sunday']
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        arrivals = []

        for day_offset in range(n_days):
            current_date = start_date + timedelta(days=day_offset)
            current_dow = current_date.weekday()
            current_month = current_date.month

            # Get seasonality multiplier for this month
            seasonal_mult = self.monthly_seasonal[
                self.monthly_seasonal['month'] == current_month
            ]['multiplier'].values[0] if len(self.monthly_seasonal) > 0 else 1.0

            # For each mode and skill combination, sample arrival count
            for _, mode_row in self.mode_mixture.iterrows():
                mode = mode_row['mode']
                mode_fraction = mode_row['fraction']

                for skill_decile in range(1, 11):
                    # Get base lambda for this (mode, skill_decile, hour)
                    mode_skill_data = self.hourly_lambda[
                        (self.hourly_lambda['mode'] == mode) &
                        (self.hourly_lambda['skill_decile'] == skill_decile)
                    ]

                    if len(mode_skill_data) == 0:
                        continue

                    # Get dow fraction for this (mode, dow)
                    dow_data = self.dow_dist[
                        (self.dow_dist['mode'] == mode) &
                        (self.dow_dist['day_of_week'] == current_dow)
                    ]
                    dow_fraction = float(dow_data['fraction'].values[0]) if len(dow_data) > 0 else (1.0 / 7)

                    for _, row in mode_skill_data.iterrows():
                        hour = int(row['hour'])
                        base_lambda = float(row['lambda'])

                        # Apply seasonality and day-of-week
                        lambda_t = base_lambda * seasonal_mult * dow_fraction

                        # Sample number of arrivals from Poisson
                        n_arrivals = rng.poisson(lambda_t)

                        # Generate each arrival
                        for _ in range(int(n_arrivals)):
                            # Add random minute/second
                            minute = rng.randint(0, 60)
                            second = rng.randint(0, 60)

                            timestamp = current_date.replace(hour=hour,
                                                            minute=minute,
                                                            second=second)

                            # Sample Elo for this arrival
                            elo = self.sample_elo(mode, skill_decile, rng)

                            arrivals.append({
                                'timestamp': timestamp,
                                'hour': hour,
                                'day_of_week': current_dow,
                                'day_name': dow_names[current_dow],
                                'mode': mode,
                                'skill_decile': skill_decile,
                                'elo': elo,
                                'month': current_month,
                                'month_name': month_names[current_month]
                            })

        df = pd.DataFrame(arrivals)

        if len(df) > 0:
            df = df.sort_values('timestamp').reset_index(drop=True)

        return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic AoE2 matchmaking arrivals'
    )
    parser.add_argument(
        '--param-dir',
        type=str,
        default='/sessions/admiring-busy-dijkstra/mnt/TOG Matchmaking/arrival_dataset',
        help='Directory with fitted parameters'
    )
    parser.add_argument(
        '--n-days',
        type=int,
        default=30,
        help='Number of days to simulate'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (ISO format, e.g., 2024-01-01)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file (default: synthetic_arrivals.csv in param-dir)'
    )

    args = parser.parse_args()

    # Generate arrivals
    generator = ArrivalGenerator(args.param_dir)
    print(f"\nGenerating synthetic arrivals for {args.n_days} days...")
    df = generator.generate(n_days=args.n_days,
                           start_date=args.start_date,
                           rng_seed=args.seed)

    # Save
    output_path = args.output or Path(args.param_dir) / 'synthetic_arrivals.csv'
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df):,} synthetic arrival events")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Saved to: {output_path}")
    print(f"\nSample:\n{df.head(10)}")
    print(f"\nSummary by mode:\n{df.groupby('mode').size()}")


if __name__ == '__main__':
    main()
