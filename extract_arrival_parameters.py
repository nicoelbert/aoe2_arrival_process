"""
Anonymized Arrival Dataset Extraction for AoE2 Matchmaking

This script extracts arrival parameters from raw matchmaking data, computing:
- Hourly Poisson arrival rates (λ) stratified by mode and skill decile
- Day-of-week arrival fractions
- Monthly seasonality multipliers
- Skill distribution parameters (mean/std Elo per decile)
- Mode mixture fractions

All data is anonymized: no player/match IDs retained, skill discretized to deciles.
"""

import sys
import csv
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import math

# Try importing pandas, numpy - with fallback
try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    HAS_PANDAS = True
except ImportError as e:
    print(f"Warning: Could not import pandas/numpy: {e}")
    print("Attempting manual implementation...")
    HAS_PANDAS = False

import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path):
    """Load matchmaking data and prepare for arrival analysis."""
    print("Loading data...")
    df = pd.read_csv(csv_path, index_col=0)

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['datetime'].dt.month
    df['date'] = df['datetime'].dt.date

    # Get player Elo (prefer p_elo over p_solo_elo when available)
    df['player_elo'] = df['p_elo'].fillna(df['p_solo_elo'])

    # Remove rows without Elo data
    df = df.dropna(subset=['player_elo'])

    # Create skill deciles (1-10, based on Elo)
    df['skill_decile'] = pd.qcut(df['player_elo'], q=10, labels=False, duplicates='drop') + 1

    print(f"Loaded {len(df):,} records from {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Modes: {df['mode'].unique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def extract_hourly_poisson_lambda(df, output_path):
    """
    Extract hourly Poisson arrival rates λ by mode and skill decile.

    Returns a CSV with columns: hour, mode, skill_decile, lambda, count, days_observed
    """
    print("\nExtracting hourly Poisson arrival rates...")

    # Count unique dates (for rate normalization)
    unique_dates = df['date'].nunique()
    print(f"  Unique dates in dataset: {unique_dates}")

    # For each (hour, mode, skill_decile), count arrivals and estimate λ
    hourly = (df.groupby(['hour', 'mode', 'skill_decile'])
              .size()
              .reset_index(name='count'))

    # Calculate λ as arrivals per day (average rate)
    hourly['lambda'] = hourly['count'] / unique_dates
    hourly['days_observed'] = unique_dates

    # Add zero-counts for missing (hour, mode, skill_decile) combinations
    hours = range(24)
    modes = df['mode'].unique()
    deciles = range(1, 11)

    idx = pd.MultiIndex.from_product([hours, modes, deciles],
                                      names=['hour', 'mode', 'skill_decile'])
    full_grid = pd.DataFrame(index=idx).reset_index()

    hourly_full = full_grid.merge(hourly, on=['hour', 'mode', 'skill_decile'], how='left')
    hourly_full['count'] = hourly_full['count'].fillna(0).astype(int)
    hourly_full['lambda'] = hourly_full['lambda'].fillna(0.0)
    hourly_full['days_observed'] = unique_dates

    # Sort for readability
    hourly_full = hourly_full.sort_values(['mode', 'hour', 'skill_decile']).reset_index(drop=True)

    hourly_full.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Grid size: {len(hourly_full)} rows")
    print(f"  Sample:\n{hourly_full.head(12)}")

    return hourly_full


def extract_dow_distribution(df, output_path):
    """
    Extract day-of-week arrival fractions by mode.

    Returns a CSV with columns: dow, mode, fraction, count
    """
    print("\nExtracting day-of-week distribution...")

    # Count arrivals by dow and mode
    dow_counts = (df.groupby(['day_of_week', 'mode'])
                  .size()
                  .reset_index(name='count'))

    # Normalize to fractions within each mode
    dow_counts['total'] = dow_counts.groupby('mode')['count'].transform('sum')
    dow_counts['fraction'] = dow_counts['count'] / dow_counts['total']

    # Map day names
    dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    dow_counts['day_name'] = dow_counts['day_of_week'].map(dow_names)

    result = dow_counts[['day_of_week', 'day_name', 'mode', 'fraction', 'count']].copy()
    result = result.sort_values(['mode', 'day_of_week']).reset_index(drop=True)

    result.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Sample:\n{result.head(7)}")

    return result


def extract_monthly_seasonality(df, output_path):
    """
    Extract monthly seasonality multipliers relative to yearly average.

    Returns a CSV with columns: month, month_name, multiplier, count
    """
    print("\nExtracting monthly seasonality...")

    # Count arrivals by month
    monthly_counts = (df.groupby('month')
                      .size()
                      .reset_index(name='count'))

    yearly_avg = monthly_counts['count'].mean()
    monthly_counts['multiplier'] = monthly_counts['count'] / yearly_avg

    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                   5: 'May', 6: 'June', 7: 'July', 8: 'August',
                   9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    monthly_counts['month_name'] = monthly_counts['month'].map(month_names)

    result = monthly_counts[['month', 'month_name', 'multiplier', 'count']].copy()
    result = result.sort_values('month').reset_index(drop=True)

    result.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Summary: mean multiplier = {result['multiplier'].mean():.3f}, "
          f"std = {result['multiplier'].std():.3f}")
    print(f"  Sample:\n{result}")

    return result


def extract_skill_distribution(df, output_path):
    """
    Extract Elo mean/std per skill decile per mode.

    Returns a CSV with columns: mode, skill_decile, count, mean_elo, std_elo,
    min_elo, q25_elo, median_elo, q75_elo, max_elo
    """
    print("\nExtracting skill distribution parameters...")

    skill_stats = (df.groupby(['mode', 'skill_decile'])['player_elo']
                   .agg(['count', 'mean', 'std', 'min',
                         ('q25', lambda x: x.quantile(0.25)),
                         ('median', lambda x: x.quantile(0.50)),
                         ('q75', lambda x: x.quantile(0.75)),
                         'max'])
                   .reset_index())

    skill_stats.columns = ['mode', 'skill_decile', 'count', 'mean_elo', 'std_elo',
                           'min_elo', 'q25_elo', 'median_elo', 'q75_elo', 'max_elo']

    # Round for readability
    for col in ['mean_elo', 'std_elo', 'min_elo', 'q25_elo', 'median_elo', 'q75_elo', 'max_elo']:
        skill_stats[col] = skill_stats[col].round(1)

    skill_stats = skill_stats.sort_values(['mode', 'skill_decile']).reset_index(drop=True)

    skill_stats.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Sample (first 3 modes):\n{skill_stats.head(30)}")

    return skill_stats


def extract_mode_mixture(df, output_path):
    """
    Extract overall fraction of arrivals per mode.

    Returns a CSV with columns: mode, count, fraction
    """
    print("\nExtracting mode mixture...")

    mode_counts = (df.groupby('mode')
                   .size()
                   .reset_index(name='count'))

    mode_counts['total'] = mode_counts['count'].sum()
    mode_counts['fraction'] = mode_counts['count'] / mode_counts['total']
    mode_counts['percentage'] = (mode_counts['fraction'] * 100).round(2)

    result = mode_counts[['mode', 'count', 'fraction', 'percentage']].copy()
    result = result.sort_values('fraction', ascending=False).reset_index(drop=True)

    result.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Summary:\n{result}")

    return result


def generate_summary_stats(df, output_path):
    """Generate summary statistics about the dataset."""
    print("\nGenerating summary statistics...")

    summary = {
        'metric': [
            'Total records',
            'Unique matches',
            'Unique players (approx)',
            'Date range (days)',
            'Modes',
            'Skill range (Elo)',
            'Mean Elo',
            'Median Elo',
            'Std Elo'
        ],
        'value': [
            f"{len(df):,}",
            f"{df['match_id'].nunique():,}",
            '(anonymized)',
            f"{(df['date'].max() - df['date'].min()).days}",
            ', '.join(sorted(df['mode'].unique())),
            f"{df['player_elo'].min():.0f} - {df['player_elo'].max():.0f}",
            f"{df['player_elo'].mean():.1f}",
            f"{df['player_elo'].median():.1f}",
            f"{df['player_elo'].std():.1f}"
        ]
    }

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_path, index=False)

    print(f"  Saved to {output_path}")
    print(f"  Summary:\n{summary_df.to_string(index=False)}")

    return summary_df


def main():
    # Paths
    input_path = Path('/sessions/admiring-busy-dijkstra/mnt/TOG Matchmaking/aoeFamiliarity/data/long_matches.csv')
    output_dir = Path('/sessions/admiring-busy-dijkstra/mnt/TOG Matchmaking/arrival_dataset')

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = load_and_prepare_data(input_path)

    # Extract parameters
    extract_hourly_poisson_lambda(df, output_dir / 'hourly_poisson_lambda.csv')
    extract_dow_distribution(df, output_dir / 'dow_distribution.csv')
    extract_monthly_seasonality(df, output_dir / 'monthly_seasonality.csv')
    extract_skill_distribution(df, output_dir / 'skill_distribution_params.csv')
    extract_mode_mixture(df, output_dir / 'mode_mixture.csv')
    generate_summary_stats(df, output_dir / 'summary_stats.csv')

    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.csv')):
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == '__main__':
    main()
