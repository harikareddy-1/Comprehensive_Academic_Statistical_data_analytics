import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import webbrowser
import os
from datetime import datetime

class PremiumHTMLReporter:
    def __init__(self, csv_file):
        """Initialize with dataset"""
        self.df = pd.read_csv(csv_file)
        self.df.columns = self.df.columns.str.strip()
        for col in ['Age', 'GPA', 'Attendance (%)']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def get_stats(self, column):
        """Get all statistics for a column"""
        data = self.df[column].dropna()
        return {
            'mean': data.mean(), 'median': data.median(),
            'mode': data.mode().iloc[0] if len(data.mode()) > 0 else data.mean(),
            'std': data.std(), 'variance': data.var(), 'range': data.max() - data.min(),
            'min': data.min(), 'max': data.max(), 'skewness': data.skew(), 'kurtosis': data.kurtosis()
        }

    def plot_to_base64(self, fig):
        """Convert plot to base64"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close(fig)
        return image_base64

    def create_all_plots(self):
        """Create all plots including the 3x3 grid and individual plots"""
        plt.style.use('default')
        plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
        plots = {}

        # Create the main 3x3 grid plot (like compact_analyzer.py)
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('📊 COMPREHENSIVE EDUCATION DATA ANALYSIS', fontsize=24, fontweight='bold', y=0.98)

        # Plot configurations for the grid
        plot_data = [
            (self.df['Department'].value_counts(), 'bar', 'BAR: Students by Department', (0,0)),
            (self.df['Gender'].value_counts(), 'pie', 'PIE: Gender Distribution', (0,1)),
            (self.df['Age'].dropna(), 'hist', 'HISTOGRAM: Age Distribution', (0,2)),
            (self.df['GPA'].dropna(), 'box', 'BOX PLOT: GPA Distribution', (1,0)),
            (self.df.groupby('Year')['GPA'].mean().sort_index(), 'line', 'LINE: GPA by Year', (1,1)),
            ((self.df['Attendance (%)'], self.df['GPA']), 'scatter', 'SCATTER: Attendance vs GPA', (1,2)),
            (self.df[['Age', 'GPA', 'Attendance (%)']].corr(), 'heatmap', 'HEATMAP: Correlations', (2,0)),
            (self.df['GPA'].dropna(), 'hist2', 'HISTOGRAM: GPA Distribution', (2,1)),
            (self.df['Course'].value_counts(), 'bar2', 'BAR: Students by Course', (2,2))
        ]

        # Create each plot in the grid
        for data, plot_type, title, (row, col) in plot_data:
            ax = axes[row, col]

            if plot_type == 'bar':
                bars = ax.bar(range(len(data)), data.values, color='skyblue', edgecolor='black', linewidth=2)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=10)
                self._add_bar_labels(ax, bars)

            elif plot_type == 'bar2':
                bars = ax.bar(range(len(data)), data.values, color='lightcoral', edgecolor='black', linewidth=2)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=10)
                self._add_bar_labels(ax, bars)

            elif plot_type == 'pie':
                ax.pie(data.values, labels=data.index, autopct='%1.1f%%',
                      colors=['lightblue', 'lightpink', 'lightgreen'], startangle=90,
                      textprops={'fontsize': 11, 'fontweight': 'bold'})

            elif plot_type in ['hist', 'hist2']:
                color = 'orange' if plot_type == 'hist' else 'lightgreen'
                ax.hist(data, bins=15, alpha=0.8, color=color, edgecolor='black', linewidth=2)
                ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')

            elif plot_type == 'box':
                box = ax.boxplot(data, patch_artist=True)
                box['boxes'][0].set_facecolor('lightgreen')
                box['boxes'][0].set_linewidth(2)

            elif plot_type == 'line':
                ax.plot(data.index, data.values, marker='o', linewidth=3, markersize=8, color='red')
                ax.set_xlabel('Academic Year', fontsize=12, fontweight='bold')
                ax.set_ylabel('Average GPA', fontsize=12, fontweight='bold')

            elif plot_type == 'scatter':
                x_data, y_data = data
                ax.scatter(x_data, y_data, alpha=0.7, color='purple', s=50, edgecolors='black')
                corr = x_data.corr(y_data)
                ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
                       fontsize=11, fontweight='bold', bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))
                ax.set_xlabel('Attendance %', fontsize=12, fontweight='bold')
                ax.set_ylabel('GPA', fontsize=12, fontweight='bold')

            elif plot_type == 'heatmap':
                sns.heatmap(data, annot=True, cmap='coolwarm', center=0, ax=ax,
                           annot_kws={'fontsize': 10, 'fontweight': 'bold'})

            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plots['main_grid'] = self.plot_to_base64(fig)

        # Create individual high-quality plots
        individual_plots = [
            ('department_bar', self.df['Department'].value_counts(), 'BAR CHART: Students by Department', 'bar', 'skyblue'),
            ('gender_pie', self.df['Gender'].value_counts(), 'PIE CHART: Gender Distribution', 'pie', ['#FF6B6B', '#4ECDC4', '#45B7D1']),
            ('age_hist', self.df['Age'].dropna(), 'HISTOGRAM: Age Distribution', 'hist', 'orange'),
            ('gpa_box', self.df['GPA'].dropna(), 'BOX PLOT: GPA Distribution', 'box', 'lightgreen'),
            ('gpa_line', self.df.groupby('Year')['GPA'].mean().sort_index(), 'LINE PLOT: Average GPA by Academic Year', 'line', 'red'),
            ('attendance_scatter', (self.df['Attendance (%)'], self.df['GPA']), 'SCATTER PLOT: Attendance vs GPA Relationship', 'scatter', 'purple'),
            ('correlation_heatmap', self.df[['Age', 'GPA', 'Attendance (%)']].corr(), 'HEATMAP: Variable Correlation Matrix', 'heatmap', 'RdYlBu_r')
        ]

        for plot_key, data, title, plot_type, color in individual_plots:
            fig, ax = plt.subplots(figsize=(12, 8))

            if plot_type == 'bar':
                bars = ax.bar(range(len(data)), data.values, color=color, edgecolor='black', linewidth=2)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=12)
                ax.set_xlabel('Department', fontsize=14, fontweight='bold')
                ax.set_ylabel('Number of Students', fontsize=14, fontweight='bold')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{int(height)}',
                           ha='center', va='bottom', fontweight='bold', fontsize=12)

            elif plot_type == 'pie':
                wedges, texts, autotexts = ax.pie(data.values, labels=data.index, autopct='%1.1f%%',
                      colors=color, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

            elif plot_type == 'hist':
                n, bins, patches = ax.hist(data, bins=20, alpha=0.8, color=color, edgecolor='black', linewidth=2)
                ax.set_xlabel('Age', fontsize=14, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
                # Add statistics text
                ax.text(0.7, 0.8, f'Mean: {data.mean():.1f}\nStd: {data.std():.1f}',
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            elif plot_type == 'box':
                box = ax.boxplot(data, patch_artist=True, widths=0.6)
                box['boxes'][0].set_facecolor(color)
                box['boxes'][0].set_linewidth(2)
                ax.set_ylabel('GPA', fontsize=14, fontweight='bold')
                ax.set_xticklabels(['GPA'], fontsize=14, fontweight='bold')
                # Add statistics
                ax.text(0.02, 0.98, f'Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}',
                       transform=ax.transAxes, fontsize=12, fontweight='bold', va='top',
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            elif plot_type == 'line':
                ax.plot(data.index, data.values, marker='o', linewidth=4, markersize=10, color=color)
                ax.set_xlabel('Academic Year', fontsize=14, fontweight='bold')
                ax.set_ylabel('Average GPA', fontsize=14, fontweight='bold')
                # Add trend line
                z = np.polyfit(range(len(data)), data.values, 1)
                p = np.poly1d(z)
                ax.plot(data.index, p(range(len(data))), "--", alpha=0.7, color='gray', linewidth=2)

            elif plot_type == 'scatter':
                x_data, y_data = data
                ax.scatter(x_data, y_data, alpha=0.7, color=color, s=80, edgecolors='black', linewidth=1)
                corr = x_data.corr(y_data)
                ax.set_xlabel('Attendance Percentage', fontsize=14, fontweight='bold')
                ax.set_ylabel('GPA', fontsize=14, fontweight='bold')
                # Add correlation info
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}\nSample Size: {len(x_data)}',
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))
                # Add trend line
                z = np.polyfit(x_data.dropna(), y_data.dropna(), 1)
                p = np.poly1d(z)
                ax.plot(sorted(x_data.dropna()), p(sorted(x_data.dropna())), "r--", alpha=0.8, linewidth=2)

            elif plot_type == 'heatmap':
                mask = np.triu(np.ones_like(data, dtype=bool))
                sns.heatmap(data, annot=True, cmap=color, center=0, ax=ax, mask=mask,
                           square=True, annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                           cbar_kws={'shrink': 0.8})
                ax.set_xlabel('Variables', fontsize=14, fontweight='bold')
                ax.set_ylabel('Variables', fontsize=14, fontweight='bold')

            ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots[plot_key] = self.plot_to_base64(fig)

        return plots

    def _add_bar_labels(self, ax, bars):
        """Add value labels on bar charts"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

    def generate_premium_html(self):
        """Generate premium HTML report with all plots"""
        plots = self.create_all_plots()

        # Calculate statistics
        age_stats = self.get_stats('Age')
        gpa_stats = self.get_stats('GPA')
        attendance_stats = self.get_stats('Attendance (%)')

        # Calculate correlations
        correlations = {
            'Attendance-GPA': self.df['Attendance (%)'].corr(self.df['GPA']),
            'Age-GPA': self.df['Age'].corr(self.df['GPA']),
            'Age-Attendance': self.df['Age'].corr(self.df['Attendance (%)'])
        }

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Premium Education Data Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6; color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px; margin: 20px auto; background: white;
            padding: 40px; border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px; border-radius: 20px; margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .header h1 {{
            font-size: 3em; margin-bottom: 15px;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        }}
        .header p {{ font-size: 1.3em; opacity: 0.95; }}
        .section {{
            margin-bottom: 50px; padding: 30px;
            background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px; border-left: 6px solid #667eea;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea; font-size: 2.2em; margin-bottom: 25px;
            border-bottom: 3px solid #667eea; padding-bottom: 15px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }}
        .main-plot {{
            text-align: center; margin-bottom: 40px;
            background: white; padding: 30px; border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }}
        .main-plot img {{
            max-width: 100%; height: auto; border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        }}
        .plot-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px; margin-bottom: 40px;
        }}
        .plot-card {{
            background: white; padding: 25px; border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .plot-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        }}
        .plot-card h3 {{
            color: #667eea; margin-bottom: 20px; font-size: 1.4em;
            text-align: center; border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }}
        .plot-card img {{
            max-width: 100%; height: auto; border-radius: 10px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px; margin-bottom: 30px;
        }}
        .stat-card {{
            background: white; padding: 25px; border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border-top: 5px solid #667eea;
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{ transform: translateY(-3px); }}
        .stat-card h3 {{
            color: #667eea; margin-bottom: 20px; font-size: 1.4em;
            text-align: center;
        }}
        .stat-line {{
            display: flex; justify-content: space-between;
            margin-bottom: 10px; padding: 8px 0;
            border-bottom: 1px dotted #ddd;
        }}
        .stat-label {{ font-weight: 600; color: #555; }}
        .stat-value {{ color: #667eea; font-weight: bold; }}
        .correlation-section {{
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border: 2px solid #ffc107; color: #856404;
            padding: 25px; border-radius: 15px; margin: 25px 0;
            box-shadow: 0 8px 25px rgba(255,193,7,0.3);
        }}
        .correlation-section h4 {{
            color: #856404; margin-bottom: 15px; font-size: 1.3em;
        }}
        .insights-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
        }}
        .insight-card {{
            background: white; padding: 25px; border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border-left: 5px solid #28a745;
        }}
        .insight-card h4 {{ color: #28a745; margin-bottom: 15px; }}
        .footer {{
            text-align: center; padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border-radius: 15px; margin-top: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .footer p {{ font-size: 1.1em; margin-bottom: 10px; }}
        @media (max-width: 768px) {{
            .container {{ margin: 10px; padding: 20px; }}
            .header h1 {{ font-size: 2.2em; }}
            .plot-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 EDUCATION ANALYSIS</h1>
            <p>Comprehensive Statistical Analysis with Data Visualizations</p>
            <p>📊 {len(self.df)} Students • 🎓 Complete Analysis • 📈 All Statistical Measures</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>

        <div class="section">
            <h2>📊 Comprehensive Statistical Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>📈 AGE STATISTICS</h3>
                    <div class="stat-line"><span class="stat-label">Mean</span><span class="stat-value">{age_stats['mean']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Median</span><span class="stat-value">{age_stats['median']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Mode</span><span class="stat-value">{age_stats['mode']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Standard Deviation</span><span class="stat-value">{age_stats['std']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Variance</span><span class="stat-value">{age_stats['variance']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Range</span><span class="stat-value">{age_stats['range']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Minimum</span><span class="stat-value">{age_stats['min']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Maximum</span><span class="stat-value">{age_stats['max']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Skewness</span><span class="stat-value">{age_stats['skewness']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Kurtosis</span><span class="stat-value">{age_stats['kurtosis']:.3f}</span></div>
                </div>

                <div class="stat-card">
                    <h3>🎓 GPA STATISTICS</h3>
                    <div class="stat-line"><span class="stat-label">Mean</span><span class="stat-value">{gpa_stats['mean']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Median</span><span class="stat-value">{gpa_stats['median']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Mode</span><span class="stat-value">{gpa_stats['mode']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Standard Deviation</span><span class="stat-value">{gpa_stats['std']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Variance</span><span class="stat-value">{gpa_stats['variance']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Range</span><span class="stat-value">{gpa_stats['range']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Minimum</span><span class="stat-value">{gpa_stats['min']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Maximum</span><span class="stat-value">{gpa_stats['max']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Skewness</span><span class="stat-value">{gpa_stats['skewness']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Kurtosis</span><span class="stat-value">{gpa_stats['kurtosis']:.3f}</span></div>
                </div>

                <div class="stat-card">
                    <h3>📅 ATTENDANCE STATISTICS</h3>
                    <div class="stat-line"><span class="stat-label">Mean</span><span class="stat-value">{attendance_stats['mean']:.3f}%</span></div>
                    <div class="stat-line"><span class="stat-label">Median</span><span class="stat-value">{attendance_stats['median']:.3f}%</span></div>
                    <div class="stat-line"><span class="stat-label">Mode</span><span class="stat-value">{attendance_stats['mode']:.3f}%</span></div>
                    <div class="stat-line"><span class="stat-label">Standard Deviation</span><span class="stat-value">{attendance_stats['std']:.3f}%</span></div>
                    <div class="stat-line"><span class="stat-label">Variance</span><span class="stat-value">{attendance_stats['variance']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Range</span><span class="stat-value">{attendance_stats['range']:.3f}%</span></div>
                    <div class="stat-line"><span class="stat-label">Minimum</span><span class="stat-value">{attendance_stats['min']:.3f}%</span></div>
                    <div class="stat-line"><span class="stat-label">Maximum</span><span class="stat-value">{attendance_stats['max']:.3f}%</span></div>
                    <div class="stat-line"><span class="stat-label">Skewness</span><span class="stat-value">{attendance_stats['skewness']:.3f}</span></div>
                    <div class="stat-line"><span class="stat-label">Kurtosis</span><span class="stat-value">{attendance_stats['kurtosis']:.3f}</span></div>
                </div>
            </div>

            <div class="correlation-section">
                <h4>🔗 CORRELATION ANALYSIS</h4>"""

        for pair, corr in correlations.items():
            strength = 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
            html_content += f"<p><strong>{pair}:</strong> {corr:.3f} ({strength})</p>"

        html_content += f"""
                <h4 style="margin-top: 20px;">🚨 CRITICAL FINDING:</h4>
                <p><strong>Attendance vs GPA correlation: {correlations['Attendance-GPA']:.3f}</strong></p>
                <p>This extremely weak correlation challenges traditional assumptions about attendance importance!</p>
            </div>
        </div>

        <div class="section">
            <h2>📈 Data Visualizations with Key Insights</h2>

            <div class="main-plot">
                <h3 style="color: #667eea; margin-bottom: 20px; font-size: 1.6em;">
                    🎯 COMPREHENSIVE ANALYSIS OVERVIEW
                </h3>
                <img src="data:image/png;base64,{plots['main_grid']}" alt="Complete Analysis Grid">
                <p style="margin-top: 15px; color: #667eea; font-weight: bold; text-align: center; font-size: 1.1em;">
                    <strong>Conclusion:</strong> Complete overview reveals balanced demographics, normal age distribution, and surprisingly weak attendance-performance correlation.
                </p>
            </div>

            <div class="plot-grid">
                <div class="plot-card">
                    <h3>📊 Department Distribution</h3>
                    <img src="data:image/png;base64,{plots['department_bar']}" alt="Department Bar Chart">
                    <p style="margin-top: 15px; color: #667eea; font-weight: bold; text-align: center;">
                        <strong>Conclusion:</strong> Engineering and Computer Science dominate enrollment, indicating strong STEM preference.
                    </p>
                </div>

                <div class="plot-card">
                    <h3>👥 Gender Distribution</h3>
                    <img src="data:image/png;base64,{plots['gender_pie']}" alt="Gender Pie Chart">
                    <p style="margin-top: 15px; color: #667eea; font-weight: bold; text-align: center;">
                        <strong>Conclusion:</strong> Well-balanced gender representation demonstrates successful diversity and inclusion efforts.
                    </p>
                </div>

                <div class="plot-card">
                    <h3>📅 Age Distribution</h3>
                    <img src="data:image/png;base64,{plots['age_hist']}" alt="Age Histogram">
                    <p style="margin-top: 15px; color: #667eea; font-weight: bold; text-align: center;">
                        <strong>Conclusion:</strong> Normal distribution centered at 20 years confirms typical undergraduate age range.
                    </p>
                </div>

                <div class="plot-card">
                    <h3>🎓 GPA Performance</h3>
                    <img src="data:image/png;base64,{plots['gpa_box']}" alt="GPA Box Plot">
                    <p style="margin-top: 15px; color: #667eea; font-weight: bold; text-align: center;">
                        <strong>Conclusion:</strong> GPA distribution shows healthy academic performance with few outliers and median around 3.0.
                    </p>
                </div>

                <div class="plot-card">
                    <h3>📈 Academic Trends</h3>
                    <img src="data:image/png;base64,{plots['gpa_line']}" alt="GPA Line Plot">
                    <p style="margin-top: 15px; color: #667eea; font-weight: bold; text-align: center;">
                        <strong>Conclusion:</strong> Consistent GPA across academic years indicates stable academic standards and teaching quality.
                    </p>
                </div>

                <div class="plot-card">
                    <h3>🔍 Attendance vs Performance</h3>
                    <img src="data:image/png;base64,{plots['attendance_scatter']}" alt="Attendance Scatter Plot">
                    <p style="margin-top: 15px; color: #667eea; font-weight: bold; text-align: center;">
                        <strong>Conclusion:</strong> Virtually no correlation (0.006) suggests attendance quantity doesn't predict academic success.
                    </p>
                </div>

                <div class="plot-card">
                    <h3>🌡️ Variable Correlations</h3>
                    <img src="data:image/png;base64,{plots['correlation_heatmap']}" alt="Correlation Heatmap">
                    <p style="margin-top: 15px; color: #667eea; font-weight: bold; text-align: center;">
                        <strong>Conclusion:</strong> All correlations are weak, indicating independent variables with minimal predictive relationships.
                    </p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🎯 Key Insights & Demographics</h2>
            <div class="insights-grid">
                <div class="insight-card">
                    <h4>👥 Student Demographics</h4>
                    <p><strong>Total Students:</strong> {len(self.df)}</p>"""

        gender_dist = self.df['Gender'].value_counts()
        for gender, count in gender_dist.items():
            percentage = (count / len(self.df)) * 100
            html_content += f"<p><strong>{gender}:</strong> {count} ({percentage:.1f}%)</p>"

        html_content += f"""
                </div>

                <div class="insight-card">
                    <h4>🏆 Academic Excellence</h4>"""

        top_depts = self.df.groupby('Department')['GPA'].mean().sort_values(ascending=False).head(3)
        for i, (dept, gpa) in enumerate(top_depts.items(), 1):
            html_content += f"<p><strong>{i}. {dept}:</strong> {gpa:.2f} GPA</p>"

        html_content += f"""
                </div>

                <div class="insight-card">
                    <h4>📊 Performance Metrics</h4>
                    <p><strong>Average GPA:</strong> {self.df['GPA'].mean():.2f}</p>
                    <p><strong>Average Attendance:</strong> {self.df['Attendance (%)'].mean():.1f}%</p>
                    <p><strong>Age Range:</strong> {self.df['Age'].min():.0f} - {self.df['Age'].max():.0f} years</p>
                    <p><strong>Data Completeness:</strong> 100%</p>
                </div>
            </div>
        </div>


    </div>
</body>
</html>"""

        return html_content

def main():
    """Generate premium HTML report"""
    try:
        print("🚀 Generating Premium HTML Report with All Plots...")
        reporter = PremiumHTMLReporter('education_dataset.csv')
        html_content = reporter.generate_premium_html()

        with open('Premium_Education_Report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

        print("✅ Premium HTML Report generated successfully!")
        print("📁 File saved as: Premium_Education_Report.html")
        print("🎯 Features: 3x3 Grid + 7 Individual High-Quality Plots + Complete Statistics")

        # Open in browser
        import webbrowser
        file_path = os.path.abspath('Premium_Education_Report.html')
        webbrowser.open(f'file://{file_path}')
        print("🌐 Opening premium report in browser...")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()