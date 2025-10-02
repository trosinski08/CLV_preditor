"""
Revenue Visualization for Banking CLV Analysis
===========================================

This script creates comprehensive visualizations of customer revenue/profit distribution,
balance trends, and CLV potential indicators for the banking customer dataset.

Author: Banking CLV Analysis Team
Date: October 2, 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

class BankingRevenueAnalyzer:
    """
    Comprehensive revenue analysis and visualization for banking customers
    """
    
    def __init__(self, data_path=None):
        """Initialize the analyzer with banking data"""
        if data_path is None:
            self.data_path = Path("data/banking_clv_cleaned.csv")
        else:
            self.data_path = Path(data_path)
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Color palette for consistent visualizations
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'dark': '#7f7f7f',
            'light': '#bcbd22',
            'accent': '#17becf'
        }
        
    def load_data(self):
        """Load and prepare the banking data"""
        print("📊 Loading banking customer data...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df):,} customer records")
            
            # Calculate additional revenue metrics
            self._calculate_revenue_metrics()
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: Could not find data file at {self.data_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _calculate_revenue_metrics(self):
        """Calculate additional revenue and profitability metrics"""
        
        # Estimated annual revenue per customer (based on banking industry standards)
        # Account fees, interest margins, transaction fees
        self.df['estimated_annual_revenue'] = (
            self.df['Balance'] * 0.02 +  # 2% interest margin
            self.df['NumOfProducts'] * 120 +  # $120 per product annually
            np.where(self.df['HasCrCard'] == 1, 150, 0) +  # Credit card fees
            np.where(self.df['IsActiveMember'] == 1, 200, 50)  # Activity-based fees
        )
        
        # Risk-adjusted revenue (lower for potential churners)
        churn_risk_discount = np.where(self.df['Exited'] == 1, 0.3, 0.9)
        self.df['risk_adjusted_revenue'] = self.df['estimated_annual_revenue'] * churn_risk_discount
        
        # Customer profitability tiers
        revenue_quantiles = self.df['estimated_annual_revenue'].quantile([0.25, 0.5, 0.75, 0.9])
        
        conditions = [
            self.df['estimated_annual_revenue'] <= revenue_quantiles[0.25],
            (self.df['estimated_annual_revenue'] > revenue_quantiles[0.25]) & 
            (self.df['estimated_annual_revenue'] <= revenue_quantiles[0.5]),
            (self.df['estimated_annual_revenue'] > revenue_quantiles[0.5]) & 
            (self.df['estimated_annual_revenue'] <= revenue_quantiles[0.75]),
            (self.df['estimated_annual_revenue'] > revenue_quantiles[0.75]) & 
            (self.df['estimated_annual_revenue'] <= revenue_quantiles[0.9]),
            self.df['estimated_annual_revenue'] > revenue_quantiles[0.9]
        ]
        
        choices = ['Low Value', 'Medium Value', 'High Value', 'Premium', 'VIP']
        self.df['revenue_tier'] = np.select(conditions, choices, default='Medium Value')
        
        # Revenue per product
        self.df['revenue_per_product'] = self.df['estimated_annual_revenue'] / np.maximum(self.df['NumOfProducts'], 1)
        
        print("✅ Calculated additional revenue metrics")
    
    def create_revenue_distribution_plots(self):
        """Create comprehensive revenue distribution visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Revenue Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Balance Distribution
        axes[0, 0].hist(self.df['Balance'], bins=50, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        axes[0, 0].set_title('Account Balance Distribution')
        axes[0, 0].set_xlabel('Account Balance ($)')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].ticklabel_format(style='plain', axis='x')
        
        # Add statistics
        mean_balance = self.df['Balance'].mean()
        median_balance = self.df['Balance'].median()
        axes[0, 0].axvline(mean_balance, color='red', linestyle='--', label=f'Mean: ${mean_balance:,.0f}')
        axes[0, 0].axvline(median_balance, color='orange', linestyle='--', label=f'Median: ${median_balance:,.0f}')
        axes[0, 0].legend()
        
        # 2. Estimated Annual Revenue Distribution
        axes[0, 1].hist(self.df['estimated_annual_revenue'], bins=50, alpha=0.7, 
                       color=self.colors['secondary'], edgecolor='black')
        axes[0, 1].set_title('Estimated Annual Revenue Distribution')
        axes[0, 1].set_xlabel('Annual Revenue ($)')
        axes[0, 1].set_ylabel('Number of Customers')
        
        # Add statistics
        mean_revenue = self.df['estimated_annual_revenue'].mean()
        median_revenue = self.df['estimated_annual_revenue'].median()
        axes[0, 1].axvline(mean_revenue, color='red', linestyle='--', label=f'Mean: ${mean_revenue:,.0f}')
        axes[0, 1].axvline(median_revenue, color='orange', linestyle='--', label=f'Median: ${median_revenue:,.0f}')
        axes[0, 1].legend()
        
        # 3. Revenue Tier Distribution
        tier_counts = self.df['revenue_tier'].value_counts()
        colors_tier = [self.colors['success'], self.colors['info'], self.colors['primary'], 
                      self.colors['warning'], self.colors['dark']]
        
        wedges, texts, autotexts = axes[1, 0].pie(tier_counts.values, labels=tier_counts.index, 
                                                 autopct='%1.1f%%', colors=colors_tier[:len(tier_counts)])
        axes[1, 0].set_title('Customer Revenue Tier Distribution')
        
        # 4. Revenue per Product Analysis
        revenue_per_product = self.df.groupby('NumOfProducts')['estimated_annual_revenue'].mean()
        axes[1, 1].bar(revenue_per_product.index, revenue_per_product.values, 
                      color=self.colors['accent'], alpha=0.8, edgecolor='black')
        axes[1, 1].set_title('Average Revenue by Number of Products')
        axes[1, 1].set_xlabel('Number of Products')
        axes[1, 1].set_ylabel('Average Annual Revenue ($)')
        
        # Add value labels on bars
        for i, v in enumerate(revenue_per_product.values):
            axes[1, 1].text(revenue_per_product.index[i], v + 50, f'${v:,.0f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'revenue_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show for headless environment
        
        print("✅ Created revenue distribution plots")
    
    def create_customer_segment_revenue_analysis(self):
        """Analyze revenue patterns across different customer segments"""
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Revenue by Geography', 'Revenue by Gender', 
                          'Revenue by Activity Status', 'Revenue by Credit Card Status'],
            specs=[[{"type": "box"}, {"type": "violin"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Revenue by Geography (Box plot)
        for i, country in enumerate(self.df['Geography'].unique()):
            country_data = self.df[self.df['Geography'] == country]['estimated_annual_revenue']
            fig.add_trace(
                go.Box(y=country_data, name=country, 
                      boxpoints='outliers', jitter=0.3, pointpos=-1.8),
                row=1, col=1
            )
        
        # 2. Revenue by Gender (Violin plot)
        for i, gender in enumerate(self.df['Gender'].unique()):
            gender_data = self.df[self.df['Gender'] == gender]['estimated_annual_revenue']
            fig.add_trace(
                go.Violin(y=gender_data, name=gender, side="positive" if i == 0 else "negative"),
                row=1, col=2
            )
        
        # 3. Revenue by Activity Status (Bar chart)
        activity_revenue = self.df.groupby('IsActiveMember')['estimated_annual_revenue'].agg(['mean', 'count'])
        activity_labels = ['Inactive', 'Active']
        
        fig.add_trace(
            go.Bar(x=activity_labels, y=activity_revenue['mean'], 
                  name='Avg Revenue', text=[f'${x:,.0f}' for x in activity_revenue['mean']],
                  textposition='auto'),
            row=2, col=1
        )
        
        # 4. Revenue vs Balance Scatter (colored by churn status)
        colors = ['red' if x == 1 else 'blue' for x in self.df['Exited']]
        fig.add_trace(
            go.Scatter(x=self.df['Balance'], y=self.df['estimated_annual_revenue'],
                      mode='markers', 
                      marker=dict(color=colors, opacity=0.6, size=4),
                      name='Revenue vs Balance',
                      text=[f'Customer: {i}<br>Balance: ${b:,.0f}<br>Revenue: ${r:,.0f}<br>Status: {"Churned" if e==1 else "Active"}'
                           for i, b, r, e in zip(self.df.index, self.df['Balance'], 
                                               self.df['estimated_annual_revenue'], self.df['Exited'])],
                      hovertemplate='%{text}<extra></extra>'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Customer Segment Revenue Analysis",
            title_x=0.5,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Geography", row=1, col=1)
        fig.update_yaxes(title_text="Annual Revenue ($)", row=1, col=1)
        fig.update_xaxes(title_text="Gender", row=1, col=2)
        fig.update_yaxes(title_text="Annual Revenue ($)", row=1, col=2)
        fig.update_xaxes(title_text="Activity Status", row=2, col=1)
        fig.update_yaxes(title_text="Average Revenue ($)", row=2, col=1)
        fig.update_xaxes(title_text="Account Balance ($)", row=2, col=2)
        fig.update_yaxes(title_text="Annual Revenue ($)", row=2, col=2)
        
        # Save interactive plot
        fig.write_html(self.results_dir / 'customer_segment_revenue_analysis.html')
        # Don't show in headless environment
        
        print("✅ Created customer segment revenue analysis")
    
    def create_clv_potential_heatmap(self):
        """Create heatmap showing CLV potential across different dimensions"""
        
        # Create pivot tables for heatmap analysis
        
        # 1. Geography vs Age Group CLV potential
        age_bins = pd.cut(self.df['Age'], bins=[0, 30, 40, 50, 60, 100], 
                         labels=['<30', '30-40', '40-50', '50-60', '60+'])
        
        geo_age_revenue = self.df.groupby(['Geography', age_bins])['estimated_annual_revenue'].mean().unstack()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CLV Potential Analysis Heatmaps', fontsize=16, fontweight='bold')
        
        # Heatmap 1: Geography vs Age
        sns.heatmap(geo_age_revenue, annot=True, fmt=',.0f', cmap='YlOrRd', 
                   ax=axes[0, 0], cbar_kws={'label': 'Average Annual Revenue ($)'})
        axes[0, 0].set_title('Revenue Potential: Geography vs Age Group')
        axes[0, 0].set_xlabel('Age Group')
        axes[0, 0].set_ylabel('Geography')
        
        # Heatmap 2: Number of Products vs Credit Score
        credit_bins = pd.cut(self.df['CreditScore'], bins=[0, 600, 700, 800, 900], 
                           labels=['Poor (<600)', 'Fair (600-700)', 'Good (700-800)', 'Excellent (800+)'])
        
        products_credit_revenue = self.df.groupby(['NumOfProducts', credit_bins])['estimated_annual_revenue'].mean().unstack()
        
        sns.heatmap(products_credit_revenue, annot=True, fmt=',.0f', cmap='Greens', 
                   ax=axes[0, 1], cbar_kws={'label': 'Average Annual Revenue ($)'})
        axes[0, 1].set_title('Revenue Potential: Products vs Credit Score')
        axes[0, 1].set_xlabel('Credit Score Range')
        axes[0, 1].set_ylabel('Number of Products')
        
        # Heatmap 3: Tenure vs Balance Quartiles
        tenure_bins = pd.cut(self.df['Tenure'], bins=[0, 2, 5, 8, 10], 
                           labels=['New (0-2y)', 'Growing (2-5y)', 'Mature (5-8y)', 'Loyal (8-10y)'])
        
        # Handle balance quartiles with potential duplicate edges
        try:
            balance_quartiles = pd.qcut(self.df['Balance'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        except ValueError:
            # If quartiles fail due to duplicates, use manual bins
            balance_max = self.df['Balance'].max()
            balance_min = self.df['Balance'].min()
            balance_bins = [balance_min, balance_max * 0.25, balance_max * 0.5, balance_max * 0.75, balance_max]
            balance_quartiles = pd.cut(self.df['Balance'], bins=balance_bins, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], include_lowest=True)
        
        tenure_balance_revenue = self.df.groupby([tenure_bins, balance_quartiles])['estimated_annual_revenue'].mean().unstack()
        
        sns.heatmap(tenure_balance_revenue, annot=True, fmt=',.0f', cmap='Blues', 
                   ax=axes[1, 0], cbar_kws={'label': 'Average Annual Revenue ($)'})
        axes[1, 0].set_title('Revenue Potential: Tenure vs Balance Quartile')
        axes[1, 0].set_xlabel('Balance Quartile')
        axes[1, 0].set_ylabel('Tenure Group')
        
        # Heatmap 4: Revenue Tier vs Churn Risk
        churn_risk = self.df['Exited'].map({0: 'Active', 1: 'Churned'})
        
        tier_churn_count = self.df.groupby(['revenue_tier', churn_risk]).size().unstack(fill_value=0)
        tier_churn_pct = tier_churn_count.div(tier_churn_count.sum(axis=1), axis=0) * 100
        
        sns.heatmap(tier_churn_pct, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   ax=axes[1, 1], cbar_kws={'label': 'Percentage (%)'})
        axes[1, 1].set_title('Churn Risk by Revenue Tier')
        axes[1, 1].set_xlabel('Customer Status')
        axes[1, 1].set_ylabel('Revenue Tier')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'clv_potential_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show for headless environment
        
        print("✅ Created CLV potential heatmaps")
    
    def generate_revenue_insights_report(self):
        """Generate comprehensive revenue insights report"""
        
        report_content = [
            "# Banking Customer Revenue Analysis Report",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"This report analyzes revenue patterns and CLV potential for {len(self.df):,} banking customers.",
            "",
            "## Key Revenue Metrics",
            "",
            f"- **Total Customer Base**: {len(self.df):,} customers",
            f"- **Average Account Balance**: ${self.df['Balance'].mean():,.2f}",
            f"- **Median Account Balance**: ${self.df['Balance'].median():,.2f}",
            f"- **Average Estimated Annual Revenue**: ${self.df['estimated_annual_revenue'].mean():,.2f}",
            f"- **Total Estimated Annual Revenue**: ${self.df['estimated_annual_revenue'].sum():,.2f}",
            "",
            "## Revenue Distribution by Tier",
            ""
        ]
        
        # Revenue tier analysis
        tier_analysis = self.df.groupby('revenue_tier').agg({
            'CustomerId': 'count',
            'estimated_annual_revenue': ['mean', 'sum'],
            'Balance': 'mean',
            'NumOfProducts': 'mean',
            'Exited': lambda x: (x == 1).mean() * 100
        }).round(2)
        
        tier_analysis.columns = ['Customer_Count', 'Avg_Revenue', 'Total_Revenue', 'Avg_Balance', 'Avg_Products', 'Churn_Rate_%']
        
        for tier in tier_analysis.index:
            data = tier_analysis.loc[tier]
            report_content.extend([
                f"### {tier}",
                f"- **Customers**: {data['Customer_Count']:,} ({data['Customer_Count']/len(self.df)*100:.1f}%)",
                f"- **Average Annual Revenue**: ${data['Avg_Revenue']:,.2f}",
                f"- **Total Revenue Contribution**: ${data['Total_Revenue']:,.2f} ({data['Total_Revenue']/self.df['estimated_annual_revenue'].sum()*100:.1f}%)",
                f"- **Average Balance**: ${data['Avg_Balance']:,.2f}",
                f"- **Average Products**: {data['Avg_Products']:.1f}",
                f"- **Churn Rate**: {data['Churn_Rate_%']:.1f}%",
                ""
            ])
        
        # Geographic analysis
        geo_analysis = self.df.groupby('Geography').agg({
            'CustomerId': 'count',
            'estimated_annual_revenue': 'mean',
            'Balance': 'mean',
            'Exited': lambda x: (x == 1).mean() * 100
        }).round(2)
        
        report_content.extend([
            "## Geographic Revenue Analysis",
            ""
        ])
        
        for country in geo_analysis.index:
            data = geo_analysis.loc[country]
            report_content.extend([
                f"### {country}",
                f"- **Customers**: {data['CustomerId']:,} ({data['CustomerId']/len(self.df)*100:.1f}%)",
                f"- **Average Annual Revenue**: ${data['estimated_annual_revenue']:,.2f}",
                f"- **Average Balance**: ${data['Balance']:,.2f}",
                f"- **Churn Rate**: {data['Exited']:.1f}%",
                ""
            ])
        
        # Key insights
        highest_revenue_tier = tier_analysis['Avg_Revenue'].idxmax()
        lowest_churn_tier = tier_analysis['Churn_Rate_%'].idxmin()
        most_profitable_geo = geo_analysis['estimated_annual_revenue'].idxmax()
        
        report_content.extend([
            "## Key Business Insights",
            "",
            f"1. **Highest Revenue Tier**: {highest_revenue_tier} customers generate the highest average revenue (${tier_analysis.loc[highest_revenue_tier, 'Avg_Revenue']:,.2f})",
            f"2. **Lowest Churn Risk**: {lowest_churn_tier} customers have the lowest churn rate ({tier_analysis.loc[lowest_churn_tier, 'Churn_Rate_%']:.1f}%)",
            f"3. **Most Profitable Geography**: {most_profitable_geo} shows highest average revenue per customer (${geo_analysis.loc[most_profitable_geo, 'estimated_annual_revenue']:,.2f})",
            f"4. **Product Cross-sell Opportunity**: Customers with more products show {self.df.groupby('NumOfProducts')['estimated_annual_revenue'].mean().iloc[-1]/self.df.groupby('NumOfProducts')['estimated_annual_revenue'].mean().iloc[0]:.1f}x higher revenue",
            "",
            "## Recommendations",
            "",
            "1. **Focus on VIP and Premium customers** for retention programs",
            "2. **Develop targeted acquisition strategies** for high-value geographic markets", 
            "3. **Implement cross-selling programs** to increase products per customer",
            "4. **Create early warning systems** for high-value customer churn prevention",
            "",
            "## Visualizations Created",
            "",
            "- `revenue_distribution_analysis.png`: Comprehensive revenue distribution plots",
            "- `customer_segment_revenue_analysis.html`: Interactive segment analysis",
            "- `clv_potential_heatmaps.png`: CLV potential across multiple dimensions",
            "",
            "---",
            "*This report provides foundation for CLV modeling and customer segmentation strategies.*"
        ])
        
        # Save report
        report_path = self.results_dir / 'revenue_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"✅ Generated revenue insights report: {report_path}")
        
        return tier_analysis, geo_analysis
    
    def run_complete_analysis(self):
        """Run the complete revenue visualization analysis"""
        
        print("🚀 Starting Banking Revenue Visualization Analysis")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            return False
        
        # Generate visualizations
        print("\n📊 Creating revenue distribution plots...")
        self.create_revenue_distribution_plots()
        
        print("\n📈 Creating customer segment revenue analysis...")
        self.create_customer_segment_revenue_analysis()
        
        print("\n🔥 Creating CLV potential heatmaps...")
        self.create_clv_potential_heatmap()
        
        print("\n📋 Generating revenue insights report...")
        tier_analysis, geo_analysis = self.generate_revenue_insights_report()
        
        # Summary statistics
        print("\n" + "=" * 50)
        print("📊 REVENUE ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Customers: {len(self.df):,}")
        print(f"Total Estimated Annual Revenue: ${self.df['estimated_annual_revenue'].sum():,.2f}")
        print(f"Average Revenue per Customer: ${self.df['estimated_annual_revenue'].mean():,.2f}")
        print(f"Revenue Range: ${self.df['estimated_annual_revenue'].min():,.2f} - ${self.df['estimated_annual_revenue'].max():,.2f}")
        print("\nTop Revenue Tier:", tier_analysis['Avg_Revenue'].idxmax())
        print("Most Profitable Geography:", geo_analysis['estimated_annual_revenue'].idxmax())
        
        print("\n✅ Revenue visualization analysis completed!")
        print(f"📁 Results saved in: {self.results_dir}")
        
        return True

def main():
    """Main execution function"""
    
    analyzer = BankingRevenueAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎉 Revenue analysis completed successfully!")
        print("📊 Ready for CLV modeling phase")
    else:
        print("\n❌ Revenue analysis failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())