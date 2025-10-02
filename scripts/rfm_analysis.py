"""
Comprehensive RFM Analysis for Banking CLV Project
Advanced customer segmentation and behavioral analysis using RFM methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class BankingRFMAnalyzer:
    """
    Professional RFM Analysis class for banking customer segmentation
    """
    
    def __init__(self, data_path="data/banking_clv_cleaned.csv"):
        """Initialize RFM analyzer with cleaned dataset"""
        self.data_path = data_path
        self.df = None
        self.rfm_df = None
        self.segments = None
        self.analysis_results = {}
        
    def load_cleaned_data(self):
        """Load the cleaned banking dataset"""
        print("📊 Loading cleaned banking dataset...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded: {len(self.df):,} customers")
            print(f"📋 Available features: {len(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def calculate_rfm_metrics(self):
        """Calculate RFM metrics for banking customers"""
        print("\n🧮 CALCULATING RFM METRICS")
        print("="*40)
        
        # Use the pre-calculated RFM components from data cleaning
        rfm_data = []
        
        for _, customer in self.df.iterrows():
            customer_metrics = {
                'CustomerId': customer['CustomerId'],
                'Recency': customer['Recency'],  # Lower = more recent
                'Frequency': customer['Frequency'],  # Number of products
                'Monetary': customer['MonetaryValue'],  # Account balance
                
                # Additional banking-specific metrics
                'Tenure': customer['Tenure'],
                'CreditScore': customer['CreditScore'],
                'IsActiveMember': customer['IsActiveMember'],
                'Geography': customer['Geography'],
                'Age': customer['Age'],
                'NumOfProducts': customer['NumOfProducts'],
                'HasCrCard': customer['HasCrCard'],
                'EstimatedSalary': customer['EstimatedSalary'],
                'Exited': customer['Exited']
            }
            rfm_data.append(customer_metrics)
        
        self.rfm_df = pd.DataFrame(rfm_data)
        
        print(f"✅ RFM metrics calculated for {len(self.rfm_df):,} customers")
        
        # Display RFM summary statistics
        print("\n📈 RFM Summary Statistics:")
        print("-" * 30)
        rfm_summary = self.rfm_df[['Recency', 'Frequency', 'Monetary']].describe()
        print(rfm_summary.round(2))
        
        return self.rfm_df
    
    def create_rfm_scores(self):
        """Create RFM scores using quintile-based scoring"""
        print("\n🎯 CREATING RFM SCORES")
        print("="*30)
        
        # Calculate quintiles for scoring
        # Note: For Recency, lower values are better (more recent)
        # For Frequency and Monetary, higher values are better
        
        # Recency Score (1-5, where 5 = most recent)
        self.rfm_df['R_Score'] = pd.qcut(self.rfm_df['Recency'], 
                                        q=5, labels=[5,4,3,2,1])
        
        # Frequency Score (1-5, where 5 = highest frequency)
        self.rfm_df['F_Score'] = pd.qcut(self.rfm_df['Frequency'].rank(method='first'), 
                                        q=5, labels=[1,2,3,4,5])
        
        # Monetary Score (1-5, where 5 = highest monetary value)
        # Handle zero values by using rank method
        self.rfm_df['M_Score'] = pd.qcut(self.rfm_df['Monetary'].rank(method='first'), 
                                        q=5, labels=[1,2,3,4,5])
        
        # Convert scores to integers
        self.rfm_df['R_Score'] = self.rfm_df['R_Score'].astype(int)
        self.rfm_df['F_Score'] = self.rfm_df['F_Score'].astype(int)
        self.rfm_df['M_Score'] = self.rfm_df['M_Score'].astype(int)
        
        # Create combined RFM Score
        self.rfm_df['RFM_Score'] = (self.rfm_df['R_Score'].astype(str) + 
                                   self.rfm_df['F_Score'].astype(str) + 
                                   self.rfm_df['M_Score'].astype(str))
        
        # Calculate RFM Score (numeric)
        self.rfm_df['RFM_Score_Numeric'] = (self.rfm_df['R_Score'] * 100 + 
                                           self.rfm_df['F_Score'] * 10 + 
                                           self.rfm_df['M_Score'])
        
        print("✅ RFM scores calculated")
        print(f"📊 Score distribution:")
        print(f"   Recency scores: {self.rfm_df['R_Score'].value_counts().sort_index().to_dict()}")
        print(f"   Frequency scores: {self.rfm_df['F_Score'].value_counts().sort_index().to_dict()}")
        print(f"   Monetary scores: {self.rfm_df['M_Score'].value_counts().sort_index().to_dict()}")
        
        return self.rfm_df
    
    def create_customer_segments(self):
        """Create customer segments based on RFM scores"""
        print("\n👥 CREATING CUSTOMER SEGMENTS")
        print("="*35)
        
        def segment_customers(row):
            """Segment customers based on RFM scores"""
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            # Champions: High value customers
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            
            # Loyal Customers: Regular users with good monetary value
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            
            # Potential Loyalists: Recent customers with potential
            elif r >= 4 and f >= 2 and m >= 2:
                return 'Potential Loyalists'
            
            # New Customers: Recent but low frequency/monetary
            elif r >= 4 and f <= 2:
                return 'New Customers'
            
            # Promising: Recent customers with some engagement
            elif r >= 3 and f >= 2 and m >= 2:
                return 'Promising'
            
            # Need Attention: Good monetary but declining activity
            elif m >= 4 and r <= 2:
                return 'Need Attention'
            
            # About to Sleep: Declining across all metrics
            elif r <= 2 and f >= 2 and m >= 2:
                return 'About to Sleep'
            
            # At Risk: Were good customers but declining
            elif r <= 2 and f >= 3 and m >= 3:
                return 'At Risk'
            
            # Cannot Lose Them: High monetary but very low recency
            elif m >= 4 and r == 1:
                return 'Cannot Lose Them'
            
            # Lost: Low across all metrics
            else:
                return 'Lost'
        
        # Apply segmentation
        self.rfm_df['Segment'] = self.rfm_df.apply(segment_customers, axis=1)
        
        # Calculate segment statistics
        segment_stats = self.rfm_df.groupby('Segment').agg({
            'CustomerId': 'count',
            'Recency': 'mean',
            'Frequency': 'mean', 
            'Monetary': 'mean',
            'IsActiveMember': 'mean',
            'Exited': 'mean'
        }).round(2)
        
        segment_stats.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 
                               'Avg_Monetary', 'Active_Rate', 'Churn_Rate']
        
        # Add percentage
        segment_stats['Percentage'] = (segment_stats['Count'] / len(self.rfm_df) * 100).round(1)
        
        # Sort by monetary value (descending)
        segment_stats = segment_stats.sort_values('Avg_Monetary', ascending=False)
        
        print("📊 Customer Segments Created:")
        print(segment_stats)
        
        self.segments = segment_stats
        return segment_stats
    
    def analyze_segment_characteristics(self):
        """Analyze detailed characteristics of each segment"""
        print("\n🔍 ANALYZING SEGMENT CHARACTERISTICS")
        print("="*45)
        
        segment_analysis = {}
        
        for segment in self.rfm_df['Segment'].unique():
            segment_data = self.rfm_df[self.rfm_df['Segment'] == segment]
            
            analysis = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(self.rfm_df) * 100,
                'avg_age': segment_data['Age'].mean(),
                'avg_tenure': segment_data['Tenure'].mean(),
                'avg_credit_score': segment_data['CreditScore'].mean(),
                'avg_salary': segment_data['EstimatedSalary'].mean(),
                'avg_balance': segment_data['Monetary'].mean(),
                'avg_products': segment_data['Frequency'].mean(),
                'active_rate': segment_data['IsActiveMember'].mean() * 100,
                'churn_rate': segment_data['Exited'].mean() * 100,
                'credit_card_rate': segment_data['HasCrCard'].mean() * 100,
                'top_countries': segment_data['Geography'].value_counts().head(3).to_dict()
            }
            
            segment_analysis[segment] = analysis
            
            print(f"\n🎯 {segment} ({analysis['size']:,} customers, {analysis['percentage']:.1f}%):")
            print(f"   💰 Avg Balance: ${analysis['avg_balance']:,.0f}")
            print(f"   🛍️  Avg Products: {analysis['avg_products']:.1f}")
            print(f"   👤 Avg Age: {analysis['avg_age']:.1f} years")
            print(f"   🏦 Avg Tenure: {analysis['avg_tenure']:.1f} years")
            print(f"   📊 Avg Credit Score: {analysis['avg_credit_score']:.0f}")
            print(f"   🔥 Active Rate: {analysis['active_rate']:.1f}%")
            print(f"   📉 Churn Rate: {analysis['churn_rate']:.1f}%")
        
        self.analysis_results['segment_analysis'] = segment_analysis
        return segment_analysis
    
    def create_rfm_visualizations(self):
        """Create comprehensive RFM visualizations"""
        print("\n📊 CREATING RFM VISUALIZATIONS")
        print("="*35)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Banking Customer RFM Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. RFM Distribution
        self.rfm_df[['Recency', 'Frequency', 'Monetary']].hist(bins=30, ax=axes[0, 0])
        axes[0, 0].set_title('RFM Metrics Distribution')
        
        # 2. Segment Size
        segment_counts = self.rfm_df['Segment'].value_counts()
        axes[0, 1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Customer Segment Distribution')
        
        # 3. RFM Scores Heatmap
        rfm_scores = self.rfm_df[['R_Score', 'F_Score', 'M_Score']].corr()
        sns.heatmap(rfm_scores, annot=True, cmap='coolwarm', ax=axes[0, 2])
        axes[0, 2].set_title('RFM Scores Correlation')
        
        # 4. Segment vs Monetary Value
        segment_monetary = self.rfm_df.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
        segment_monetary.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Average Monetary Value by Segment')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Churn Rate by Segment
        churn_by_segment = self.rfm_df.groupby('Segment')['Exited'].mean() * 100
        churn_by_segment.plot(kind='bar', ax=axes[1, 1], color='coral')
        axes[1, 1].set_title('Churn Rate by Segment (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. RFM 3D Scatter (using 2D representation)
        scatter = axes[1, 2].scatter(self.rfm_df['Frequency'], self.rfm_df['Monetary'], 
                                   c=self.rfm_df['Recency'], alpha=0.6, cmap='viridis')
        axes[1, 2].set_xlabel('Frequency (Products)')
        axes[1, 2].set_ylabel('Monetary (Balance)')
        axes[1, 2].set_title('Frequency vs Monetary (colored by Recency)')
        plt.colorbar(scatter, ax=axes[1, 2])
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('results/rfm_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print("✅ RFM Dashboard saved: results/rfm_analysis_dashboard.png")
        
        # Show the plot
        plt.show()
        
        return fig
    
    def create_interactive_visualizations(self):
        """Create interactive Plotly visualizations"""
        print("\n🎮 CREATING INTERACTIVE VISUALIZATIONS")
        print("="*45)
        
        # 1. Interactive 3D RFM Scatter Plot
        fig_3d = px.scatter_3d(
            self.rfm_df, 
            x='Recency', y='Frequency', z='Monetary',
            color='Segment',
            title='Interactive 3D RFM Analysis',
            labels={'Recency': 'Recency Score', 
                   'Frequency': 'Frequency Score', 
                   'Monetary': 'Monetary Value'},
            hover_data=['CustomerId', 'Age', 'Tenure']
        )
        fig_3d.write_html('results/rfm_3d_interactive.html')
        print("✅ 3D Interactive plot saved: results/rfm_3d_interactive.html")
        
        # 2. Segment Performance Dashboard
        fig_dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Segment Distribution', 'Average Monetary Value', 
                          'Churn Rate by Segment', 'Activity Rate by Segment'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Pie chart for segment distribution
        segment_counts = self.rfm_df['Segment'].value_counts()
        fig_dashboard.add_trace(
            go.Pie(labels=segment_counts.index, values=segment_counts.values, name="Segments"),
            row=1, col=1
        )
        
        # Bar chart for monetary value
        segment_monetary = self.rfm_df.groupby('Segment')['Monetary'].mean()
        fig_dashboard.add_trace(
            go.Bar(x=segment_monetary.index, y=segment_monetary.values, name="Avg Monetary"),
            row=1, col=2
        )
        
        # Bar chart for churn rate
        churn_by_segment = self.rfm_df.groupby('Segment')['Exited'].mean() * 100
        fig_dashboard.add_trace(
            go.Bar(x=churn_by_segment.index, y=churn_by_segment.values, name="Churn Rate"),
            row=2, col=1
        )
        
        # Bar chart for activity rate
        activity_by_segment = self.rfm_df.groupby('Segment')['IsActiveMember'].mean() * 100
        fig_dashboard.add_trace(
            go.Bar(x=activity_by_segment.index, y=activity_by_segment.values, name="Activity Rate"),
            row=2, col=2
        )
        
        fig_dashboard.update_layout(title_text="Banking Customer Segments Performance Dashboard")
        fig_dashboard.write_html('results/segment_dashboard.html')
        print("✅ Segment dashboard saved: results/segment_dashboard.html")
        
        return fig_3d, fig_dashboard
    
    def generate_business_insights(self):
        """Generate actionable business insights from RFM analysis"""
        print("\n💡 GENERATING BUSINESS INSIGHTS")
        print("="*40)
        
        insights = []
        
        # Top segments by monetary value
        top_segments = self.segments.sort_values('Avg_Monetary', ascending=False).head(3)
        insights.append(f"🏆 Top 3 valuable segments: {', '.join(top_segments.index.tolist())}")
        
        # Churn risk analysis
        high_churn_segments = self.segments[self.segments['Churn_Rate'] > 30]
        if len(high_churn_segments) > 0:
            insights.append(f"⚠️  High churn risk segments: {', '.join(high_churn_segments.index.tolist())}")
        
        # Growth opportunities
        low_frequency_segments = self.segments[self.segments['Avg_Frequency'] < 2]
        if len(low_frequency_segments) > 0:
            insights.append(f"📈 Cross-selling opportunities: {', '.join(low_frequency_segments.index.tolist())}")
        
        # Champions analysis
        champions = self.rfm_df[self.rfm_df['Segment'] == 'Champions']
        if len(champions) > 0:
            insights.append(f"👑 Champions represent {len(champions)/len(self.rfm_df)*100:.1f}% but contribute ${champions['Monetary'].sum():,.0f} in total balance")
        
        # At-risk customers
        at_risk = self.rfm_df[self.rfm_df['Segment'].isin(['At Risk', 'Cannot Lose Them', 'About to Sleep'])]
        if len(at_risk) > 0:
            insights.append(f"🚨 {len(at_risk):,} customers need immediate retention attention")
        
        print("\\n💼 KEY BUSINESS INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        self.analysis_results['business_insights'] = insights
        return insights
    
    def save_rfm_results(self):
        """Save RFM analysis results"""
        print("\n💾 SAVING RFM ANALYSIS RESULTS")
        print("="*35)
        
        # Save RFM dataset
        self.rfm_df.to_csv('data/banking_rfm_analysis.csv', index=False)
        print("✅ RFM dataset saved: data/banking_rfm_analysis.csv")
        
        # Save segment summary
        self.segments.to_csv('results/rfm_segments_summary.csv')
        print("✅ Segments summary saved: results/rfm_segments_summary.csv")
        
        # Create comprehensive report
        report = f"""# Banking RFM Analysis Report

## Executive Summary
RFM analysis completed for {len(self.rfm_df):,} banking customers, identifying {len(self.rfm_df['Segment'].unique())} distinct customer segments.

## Segment Overview
{self.segments.to_string()}

## Key Business Insights
"""
        for i, insight in enumerate(self.analysis_results.get('business_insights', []), 1):
            report += f"{i}. {insight}\\n"
        
        report += f"""
## RFM Metrics Summary
- Average Recency: {self.rfm_df['Recency'].mean():.2f}
- Average Frequency: {self.rfm_df['Frequency'].mean():.2f}
- Average Monetary: ${self.rfm_df['Monetary'].mean():,.2f}

## Next Steps
1. Implement targeted retention campaigns for at-risk segments
2. Design cross-selling strategies for low-frequency customers
3. Create loyalty programs for Champions and Loyal Customers
4. Develop reactivation campaigns for Lost customers

---
*Analysis generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open('results/rfm_analysis_report.md', 'w') as f:
            f.write(report)
        print("✅ Comprehensive report saved: results/rfm_analysis_report.md")
    
    def run_complete_rfm_analysis(self):
        """Run the complete RFM analysis pipeline"""
        print("🚀 STARTING COMPREHENSIVE RFM ANALYSIS")
        print("="*60)
        
        # Load data
        if not self.load_cleaned_data():
            return False
        
        # Calculate RFM metrics
        self.calculate_rfm_metrics()
        
        # Create RFM scores
        self.create_rfm_scores()
        
        # Create customer segments
        self.create_customer_segments()
        
        # Analyze segment characteristics
        self.analyze_segment_characteristics()
        
        # Create visualizations
        self.create_rfm_visualizations()
        self.create_interactive_visualizations()
        
        # Generate business insights
        self.generate_business_insights()
        
        # Save results
        self.save_rfm_results()
        
        print("\\n🎉 RFM ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("📊 Results saved in data/ and results/ directories")
        print("📈 Visualizations created: PNG and HTML formats")
        print("📋 Ready for CLV modeling phase")
        
        return True

def main():
    """Main function to run RFM analysis"""
    analyzer = BankingRFMAnalyzer()
    success = analyzer.run_complete_rfm_analysis()
    
    if success:
        return analyzer
    else:
        print("❌ RFM analysis failed!")
        return None

if __name__ == "__main__":
    rfm_analyzer = main()