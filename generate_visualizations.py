"""
Visualization Generator for Arrival Dataset

Creates publication-ready visualizations:
1. Heatmap: Arrival rate (λ) by hour × day-of-week × mode
2. Skill distributions: Elo by decile and mode
3. Monthly seasonality: Multiplier trends
4. Mode mixture: Pie chart of arrival fractions
5. Temporal patterns: Hour and day distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5
})


class ArrivalVisualizer:
    """Creates visualizations for arrival dataset."""

    def __init__(self, param_dir):
        """Load parameters from directory."""
        self.param_dir = Path(param_dir)

        self.hourly_lambda = pd.read_csv(self.param_dir / 'hourly_poisson_lambda.csv')
        self.dow_dist = pd.read_csv(self.param_dir / 'dow_distribution.csv')
        self.monthly_seasonal = pd.read_csv(self.param_dir / 'monthly_seasonality.csv')
        self.skill_dist = pd.read_csv(self.param_dir / 'skill_distribution_params.csv')
        self.mode_mixture = pd.read_csv(self.param_dir / 'mode_mixture.csv')

    def plot_hourly_heatmap_by_mode(self):
        """Create heatmap of hourly arrival rate (λ) by hour × mode."""
        print("Creating hourly heatmap...")

        modes = self.hourly_lambda['mode'].unique()
        n_modes = len(modes)

        fig, axes = plt.subplots(1, n_modes, figsize=(5*n_modes, 6), squeeze=False)
        axes = axes.flatten()

        for idx, mode in enumerate(sorted(modes)):
            mode_data = self.hourly_lambda[self.hourly_lambda['mode'] == mode]

            # Aggregate across skill deciles to get hourly pattern
            hourly_agg = (mode_data.groupby('hour')['lambda']
                         .sum()
                         .reset_index()
                         .set_index('hour')
                         .reindex(range(24), fill_value=0))

            ax = axes[idx]
            bars = ax.bar(range(24), hourly_agg.values.flatten(), color='steelblue', alpha=0.7)

            # Color intensity based on rate
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    bar.set_color(plt.cm.Blues(min(height / hourly_agg.values.max(), 1.0)))

            ax.set_xlabel('Hour of Day', fontsize=10)
            ax.set_ylabel('Arrival Rate λ', fontsize=10)
            ax.set_title(f'{mode}: Hourly Arrivals', fontsize=11, fontweight='bold')
            ax.set_xticks(range(0, 24, 2))
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.param_dir / 'viz_hourly_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()

    def plot_hourly_dow_heatmap(self):
        """Create heatmap of arrival rate by hour × day-of-week for dominant mode."""
        print("Creating hour × day-of-week heatmap...")

        # Use the most common mode
        dominant_mode = self.mode_mixture.nlargest(1, 'count')['mode'].values[0]
        mode_data = self.hourly_lambda[self.hourly_lambda['mode'] == dominant_mode]

        # Aggregate across skill deciles by hour
        hourly_by_hour = mode_data.groupby('hour')['lambda'].sum().reset_index()

        # Get dow fractions for this mode
        mode_dow = self.dow_dist[self.dow_dist['mode'] == dominant_mode].copy()
        mode_dow['dow'] = mode_dow['day_of_week']

        # Create matrix: hours × days
        heatmap_data = np.zeros((24, 7))
        for hour in range(24):
            hour_rate = hourly_by_hour[hourly_by_hour['hour'] == hour]['lambda'].values
            if len(hour_rate) > 0:
                for dow in range(7):
                    dow_frac = mode_dow[mode_dow['dow'] == dow]['fraction'].values
                    if len(dow_frac) > 0:
                        heatmap_data[hour, dow] = hour_rate[0] * float(dow_frac[0])

        # Convert to DataFrame for seaborn
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        heatmap_df = pd.DataFrame(heatmap_data, columns=dow_names,
                                  index=range(24))

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(heatmap_df, cmap='YlOrRd', annot=False, fmt='.2f',
                   cbar_kws={'label': 'Arrival Rate λ'}, ax=ax, linewidths=0.5)

        ax.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
        ax.set_ylabel('Hour of Day', fontsize=11, fontweight='bold')
        ax.set_title(f'Hourly × Day-of-Week Arrival Rate: {dominant_mode}',
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_path = self.param_dir / 'viz_hourly_dow_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()

    def plot_skill_distributions(self):
        """Plot Elo distributions by skill decile and mode."""
        print("Creating skill distribution plots...")

        modes = sorted(self.skill_dist['mode'].unique())
        n_modes = len(modes)

        fig, axes = plt.subplots(n_modes, 2, figsize=(14, 4*n_modes))
        if n_modes == 1:
            axes = axes.reshape(1, -1)

        for mode_idx, mode in enumerate(modes):
            mode_data = self.skill_dist[self.skill_dist['mode'] == mode].sort_values('skill_decile')

            # Left: Mean and std by decile
            ax = axes[mode_idx, 0]
            deciles = mode_data['skill_decile'].values
            means = mode_data['mean_elo'].values
            stds = mode_data['std_elo'].values

            ax.errorbar(deciles, means, yerr=stds, fmt='o-', capsize=5,
                       color='steelblue', ecolor='lightblue', alpha=0.7, markersize=8)

            ax.fill_between(deciles, means - stds, means + stds, alpha=0.2, color='steelblue')
            ax.set_xlabel('Skill Decile', fontsize=10)
            ax.set_ylabel('Elo', fontsize=10)
            ax.set_title(f'{mode}: Elo by Decile (mean ± std)', fontsize=11, fontweight='bold')
            ax.set_xticks(range(1, 11))
            ax.grid(True, alpha=0.3)

            # Right: Box plot representation
            ax = axes[mode_idx, 1]
            decile_labels = [f'D{i}' for i in deciles]
            box_data = [mode_data[mode_data['skill_decile'] == d][['min_elo', 'q25_elo',
                                                                    'median_elo', 'q75_elo', 'max_elo']].values.flatten()
                       for d in deciles]

            bp = ax.boxplot(box_data, labels=decile_labels, patch_artist=True)

            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)

            ax.set_xlabel('Skill Decile', fontsize=10)
            ax.set_ylabel('Elo', fontsize=10)
            ax.set_title(f'{mode}: Elo Distribution by Decile', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.param_dir / 'viz_skill_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()

    def plot_monthly_seasonality(self):
        """Plot monthly seasonality multipliers."""
        print("Creating monthly seasonality plot...")

        fig, ax = plt.subplots(figsize=(12, 6))

        months = self.monthly_seasonal['month'].values
        month_names_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        multipliers = self.monthly_seasonal['multiplier'].values

        # Bar plot
        colors = ['steelblue' if m >= 1.0 else 'coral' for m in multipliers]
        ax.bar(months, multipliers, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        # Reference line at 1.0
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Average (1.0x)')

        ax.set_xlabel('Month', fontsize=11, fontweight='bold')
        ax.set_ylabel('Seasonality Multiplier', fontsize=11, fontweight='bold')
        ax.set_title('Monthly Seasonality: Relative to Yearly Average', fontsize=12, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels([month_names_short[m-1] for m in months])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.param_dir / 'viz_monthly_seasonality.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()

    def plot_mode_mixture(self):
        """Plot mode mixture fractions."""
        print("Creating mode mixture plot...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart
        ax = axes[0]
        modes = self.mode_mixture['mode'].values
        fractions = self.mode_mixture['fraction'].values
        colors = sns.color_palette('husl', len(modes))

        wedges, texts, autotexts = ax.pie(
            fractions, labels=modes, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 10}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Mode Mixture: Arrival Fractions', fontsize=11, fontweight='bold')

        # Bar chart with counts
        ax = axes[1]
        counts = self.mode_mixture['count'].values
        ax.barh(modes, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        ax.set_xlabel('Number of Arrivals', fontsize=10)
        ax.set_title('Mode Mixture: Absolute Counts', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add count labels
        for i, count in enumerate(counts):
            ax.text(count, i, f' {count:,}', va='center', fontsize=9)

        plt.tight_layout()
        output_path = self.param_dir / 'viz_mode_mixture.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()

    def plot_temporal_patterns(self):
        """Plot hour and day-of-week distributions."""
        print("Creating temporal patterns plot...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Hour distribution (all modes)
        ax = axes[0, 0]
        hourly_totals = self.hourly_lambda.groupby('hour')['lambda'].sum()
        ax.bar(range(24), hourly_totals.values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Hour of Day', fontsize=10)
        ax.set_ylabel('Total Arrival Rate λ', fontsize=10)
        ax.set_title('Hourly Distribution (All Modes)', fontsize=11, fontweight='bold')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3, axis='y')

        # Day-of-week distribution (all modes)
        ax = axes[0, 1]
        dow_totals = self.dow_dist.groupby('day_of_week')['fraction'].mean()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax.bar(range(7), dow_totals.values, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Day of Week', fontsize=10)
        ax.set_ylabel('Mean Arrival Fraction', fontsize=10)
        ax.set_title('Day-of-Week Distribution (All Modes)', fontsize=11, fontweight='bold')
        ax.set_xticks(range(7))
        ax.set_xticklabels(dow_names)
        ax.grid(True, alpha=0.3, axis='y')

        # Hour distribution by mode
        ax = axes[1, 0]
        modes = sorted(self.hourly_lambda['mode'].unique())
        for mode in modes:
            mode_data = self.hourly_lambda[self.hourly_lambda['mode'] == mode]
            hourly = mode_data.groupby('hour')['lambda'].sum()
            ax.plot(range(24), hourly.values, marker='o', label=mode, linewidth=2, markersize=4)

        ax.set_xlabel('Hour of Day', fontsize=10)
        ax.set_ylabel('Arrival Rate λ', fontsize=10)
        ax.set_title('Hourly Pattern by Mode', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)

        # Skill decile distribution
        ax = axes[1, 1]
        decile_counts = self.skill_dist.groupby('skill_decile')['count'].sum()
        ax.bar(decile_counts.index, decile_counts.values, color='lightgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Skill Decile', fontsize=10)
        ax.set_ylabel('Total Arrivals', fontsize=10)
        ax.set_title('Skill Decile Distribution (All Modes)', fontsize=11, fontweight='bold')
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.param_dir / 'viz_temporal_patterns.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()

    def generate_all(self):
        """Generate all visualizations."""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70 + "\n")

        self.plot_hourly_heatmap_by_mode()
        self.plot_hourly_dow_heatmap()
        self.plot_skill_distributions()
        self.plot_monthly_seasonality()
        self.plot_mode_mixture()
        self.plot_temporal_patterns()

        print("\n" + "="*70)
        print("VISUALIZATION COMPLETE")
        print("="*70)

        viz_files = list(self.param_dir.glob('viz_*.png'))
        print(f"\nGenerated {len(viz_files)} visualization files:")
        for f in sorted(viz_files):
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate arrival dataset visualizations')
    parser.add_argument(
        '--param-dir',
        type=str,
        default='/sessions/admiring-busy-dijkstra/mnt/TOG Matchmaking/arrival_dataset',
        help='Directory with fitted parameters'
    )

    args = parser.parse_args()

    visualizer = ArrivalVisualizer(args.param_dir)
    visualizer.generate_all()


if __name__ == '__main__':
    main()
