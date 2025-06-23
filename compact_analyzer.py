import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CompactEducationAnalyzer:
    def __init__(self, csv_file):
        """Initialize with dataset"""
        self.df = pd.read_csv(csv_file)
        self.df.columns = self.df.columns.str.strip()
        # Convert to numeric
        for col in ['Age', 'GPA', 'Attendance (%)']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        print(f"📊 Dataset loaded: {len(self.df)} students")
    
    def get_stats(self, column):
        """Get all statistics for a column"""
        data = self.df[column].dropna()
        return {
            'mean': data.mean(), 'median': data.median(), 'mode': data.mode().iloc[0] if len(data.mode()) > 0 else data.mean(),
            'std': data.std(), 'variance': data.var(), 'range': data.max() - data.min(),
            'min': data.min(), 'max': data.max(), 'skewness': data.skew(), 'kurtosis': data.kurtosis()
        }
    
    def print_stats(self):
        """Print comprehensive statistics line by line for all numeric columns"""
        print("\n📊 COMPREHENSIVE STATISTICAL SUMMARY")
        print("="*60)
        for col in ['Age', 'GPA', 'Attendance (%)']:
            stats = self.get_stats(col)
            print(f"\n📈 {col.upper()}:")
            print(f"  Mean..................: {stats['mean']:.3f}")
            print(f"  Median................: {stats['median']:.3f}")
            print(f"  Mode..................: {stats['mode']:.3f}")
            print(f"  Standard Deviation....: {stats['std']:.3f}")
            print(f"  Variance..............: {stats['variance']:.3f}")
            print(f"  Range.................: {stats['range']:.3f}")
            print(f"  Minimum...............: {stats['min']:.3f}")
            print(f"  Maximum...............: {stats['max']:.3f}")
            print(f"  Skewness..............: {stats['skewness']:.3f}")
            print(f"  Kurtosis..............: {stats['kurtosis']:.3f}")
    
    def create_all_plots(self):
        """Create all 7 required plots in one function"""
        # Setup
        plt.style.use('default')
        plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('📊 EDUCATION DATA ANALYSIS', fontsize=20, fontweight='bold')
        
        # Plot configurations: (data_func, plot_type, title, position)
        plots = [
            (lambda: self.df['Department'].value_counts(), 'bar', 'BAR: Students by Department', (0,0)),
            (lambda: self.df['Gender'].value_counts(), 'pie', 'PIE: Gender Distribution', (0,1)),
            (lambda: self.df['Age'].dropna(), 'hist', 'HISTOGRAM: Age Distribution', (0,2)),
            (lambda: self.df['GPA'].dropna(), 'box', 'BOX PLOT: GPA Distribution', (1,0)),
            (lambda: self.df.groupby('Year')['GPA'].mean().sort_index(), 'line', 'LINE: GPA by Year', (1,1)),
            (lambda: (self.df['Attendance (%)'], self.df['GPA']), 'scatter', 'SCATTER: Attendance vs GPA', (1,2)),
            (lambda: self.df[['Age', 'GPA', 'Attendance (%)']].corr(), 'heatmap', 'HEATMAP: Correlations', (2,0)),
            (lambda: self.df['GPA'].dropna(), 'hist2', 'HISTOGRAM: GPA Distribution', (2,1)),
            (lambda: self.df['Course'].value_counts(), 'bar2', 'BAR: Students by Course', (2,2))
        ]
        
        # Create each plot
        for data_func, plot_type, title, (row, col) in plots:
            ax = axes[row, col]
            data = data_func()
            
            if plot_type == 'bar':
                bars = ax.bar(range(len(data)), data.values, color='skyblue', edgecolor='black')
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(data.index, rotation=45, ha='right')
                self._add_bar_labels(ax, bars)
                
            elif plot_type == 'bar2':
                bars = ax.bar(range(len(data)), data.values, color='lightcoral', edgecolor='black')
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(data.index, rotation=45, ha='right')
                self._add_bar_labels(ax, bars)
                
            elif plot_type == 'pie':
                ax.pie(data.values, labels=data.index, autopct='%1.1f%%', 
                      colors=['lightblue', 'lightpink', 'lightgreen'], startangle=90)
                
            elif plot_type in ['hist', 'hist2']:
                color = 'orange' if plot_type == 'hist' else 'lightgreen'
                ax.hist(data, bins=15, alpha=0.8, color=color, edgecolor='black')
                ax.set_ylabel('Frequency')
                
            elif plot_type == 'box':
                box = ax.boxplot(data, patch_artist=True)
                box['boxes'][0].set_facecolor('lightgreen')
                
            elif plot_type == 'line':
                ax.plot(data.index, data.values, marker='o', linewidth=3, markersize=8, color='red')
                ax.set_xlabel('Academic Year')
                ax.set_ylabel('Average GPA')
                
            elif plot_type == 'scatter':
                x_data, y_data = data
                ax.scatter(x_data, y_data, alpha=0.7, color='purple', s=50)
                corr = x_data.corr(y_data)
                ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))
                ax.set_xlabel('Attendance %')
                ax.set_ylabel('GPA')
                
            elif plot_type == 'heatmap':
                sns.heatmap(data, annot=True, cmap='coolwarm', center=0, ax=ax,
                           annot_kws={'fontsize': 10, 'fontweight': 'bold'})
            
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('compact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Analysis saved as 'compact_analysis.png'")
    
    def _add_bar_labels(self, ax, bars):
        """Add value labels on bar charts"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    def generate_summary(self):
        """Generate key insights"""
        print("\n📋 KEY INSIGHTS")
        print("="*50)
        
        # Basic stats
        print(f"👥 Total Students: {len(self.df)}")
        print(f"🎓 Average GPA: {self.df['GPA'].mean():.2f}")
        print(f"📅 Average Attendance: {self.df['Attendance (%)'].mean():.1f}%")
        
        # Demographics
        gender_dist = self.df['Gender'].value_counts()
        print(f"\n👥 Gender Distribution:")
        for gender, count in gender_dist.items():
            print(f"   {gender}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Top departments
        top_depts = self.df.groupby('Department')['GPA'].mean().sort_values(ascending=False).head(3)
        print(f"\n🏆 Top Departments by GPA:")
        for i, (dept, gpa) in enumerate(top_depts.items(), 1):
            print(f"   {i}. {dept}: {gpa:.2f}")
        
        # Correlation analysis
        correlations = {
            'Attendance-GPA': self.df['Attendance (%)'].corr(self.df['GPA']),
            'Age-GPA': self.df['Age'].corr(self.df['GPA']),
            'Age-Attendance': self.df['Age'].corr(self.df['Attendance (%)'])
        }

        print(f"\n🔗 CORRELATIONS:")
        for pair, corr in correlations.items():
            strength = 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
            print(f"   {pair}: {corr:.3f} ({strength})")

        print(f"\n🚨 Critical Finding:")
        print(f"   Attendance vs GPA correlation: {correlations['Attendance-GPA']:.3f}")
        print(f"   This suggests attendance {'strongly' if abs(correlations['Attendance-GPA']) > 0.7 else 'weakly'} predicts performance.")
    
    def run_analysis(self):
        """Run complete analysis"""
        print("🚀 Starting Compact Education Data Analysis...")
        print("="*60)
        
        # Print statistics
        self.print_stats()
        
        # Create visualizations
        print("\n📊 Generating visualizations...")
        self.create_all_plots()
        
        # Generate summary
        self.generate_summary()
        
        print("\n🎉 Analysis Complete!")
        print("📁 Visualization saved as 'compact_analysis.png'")
        print("📊 Includes all 7 plot types: Bar, Pie, Histogram, Box, Line, Scatter, Heatmap")

def main():
    """Main function"""
    try:
        analyzer = CompactEducationAnalyzer('education_dataset.csv')
        analyzer.run_analysis()
    except FileNotFoundError:
        print("❌ Error: 'education_dataset.csv' not found!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
