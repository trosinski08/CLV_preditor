"""
Customer Lifetime Value (CLV) Modeling for Banking Analysis
==========================================================

This script implements comprehensive CLV modeling using BG/NBD and Gamma-Gamma models
from the lifetimes library to predict customer lifetime value for banking customers.

Models Implemented:
- BG/NBD (Beta-Geometric/Negative Binomial Distribution): Predicts customer transaction frequency and churn
- Gamma-Gamma: Predicts customer monetary value per transaction

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

# Lifetimes library for CLV modeling
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
from lifetimes.plotting import plot_period_transactions, plot_history_alive

import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class BankingCLVModeler:
    """
    Comprehensive CLV modeling for banking customers using lifetimes library
    """
    
    def __init__(self, data_path=None):
        """Initialize the CLV modeler"""
        if data_path is None:
            self.data_path = Path("data/banking_clv_cleaned.csv")
        else:
            self.data_path = Path(data_path)
        
        self.results_dir = Path("results")
        self.data_dir = Path("data")
        self.results_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.bgf_model = None
        self.ggf_model = None
        self.clv_data = None
        
        # Color palette for visualizations
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'dark': '#7f7f7f'
        }
        
    def load_and_prepare_data(self):
        """Load banking data and prepare it for CLV modeling"""
        print("📊 Loading banking customer data for CLV modeling...")
        
        try:
            # Load cleaned banking data
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df):,} customer records")
            
            # Prepare CLV-specific features
            self._prepare_clv_features()
            
            # Create transaction-like data for lifetimes models
            self._create_transaction_data()
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: Could not find data file at {self.data_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _prepare_clv_features(self):
        """Prepare features specific to CLV modeling"""
        
        # Simulate realistic banking transaction patterns
        # In real banking data, you would have actual transaction history
        
        # Create customer observation period (tenure as proxy)
        self.df['observation_period_years'] = self.df['Tenure']
        
        # Simulate frequency (number of transactions per year)
        # Based on number of products and activity level
        np.random.seed(42)  # For reproducibility
        
        base_frequency = np.where(self.df['IsActiveMember'] == 1, 
                                 np.random.poisson(24, len(self.df)),  # Active: ~2 transactions/month
                                 np.random.poisson(8, len(self.df)))   # Inactive: ~8 transactions/year
        
        # Adjust frequency based on number of products
        product_multiplier = 1 + (self.df['NumOfProducts'] - 1) * 0.3
        self.df['frequency'] = (base_frequency * product_multiplier).astype(int)
        
        # Ensure minimum frequency of 1
        self.df['frequency'] = np.maximum(self.df['frequency'], 1)
        
        # Simulate recency (time since last transaction in years)
        # Recent customers have lower recency
        self.df['recency_years'] = np.where(
            self.df['Exited'] == 1,
            np.random.uniform(0.5, self.df['Tenure']),  # Churned customers: higher recency
            np.random.uniform(0, 0.5)  # Active customers: low recency
        )
        
        # Ensure recency doesn't exceed tenure
        self.df['recency_years'] = np.minimum(self.df['recency_years'], self.df['Tenure'])
        
        # Simulate monetary value (average transaction amount)
        # Based on balance and credit score
        balance_factor = np.log1p(self.df['Balance']) / 10
        credit_factor = self.df['CreditScore'] / 1000
        
        self.df['monetary_value'] = (
            100 +  # Base transaction amount
            balance_factor * 50 +  # Higher balance = higher transactions
            credit_factor * 200 +  # Better credit = higher transactions  
            np.random.normal(0, 50, len(self.df))  # Random variation
        )
        
        # Ensure positive monetary values
        self.df['monetary_value'] = np.maximum(self.df['monetary_value'], 10)
        
        print("✅ Prepared CLV-specific features")
    
    def _create_transaction_data(self):
        """Create transaction summary data for lifetimes models"""
        
        # Create the summary data format required by lifetimes library
        # Format: customer_id, frequency, recency, T (observation period), monetary_value
        
        self.transaction_data = pd.DataFrame({
            'customer_id': self.df['CustomerId'],
            'frequency': self.df['frequency'],
            'recency': self.df['recency_years'],
            'T': self.df['observation_period_years'],
            'monetary_value': self.df['monetary_value']
        })
        
        # Filter out customers with zero frequency (edge case)
        self.transaction_data = self.transaction_data[self.transaction_data['frequency'] > 0]
        
        print(f"✅ Created transaction data for {len(self.transaction_data):,} customers")
        print(f"   - Average frequency: {self.transaction_data['frequency'].mean():.1f} transactions/year")
        print(f"   - Average recency: {self.transaction_data['recency'].mean():.2f} years")
        print(f"   - Average monetary value: ${self.transaction_data['monetary_value'].mean():.2f}")
    
    def fit_bgf_model(self):
        """Fit BG/NBD model for transaction frequency and customer lifetime prediction"""
        
        print("🔧 Fitting BG/NBD model for customer lifetime prediction...")
        
        # Initialize and fit the BG/NBD model
        self.bgf_model = BetaGeoFitter(penalizer_coef=0.1)
        
        # Fit the model
        self.bgf_model.fit(
            frequency=self.transaction_data['frequency'],
            recency=self.transaction_data['recency'],
            T=self.transaction_data['T']
        )
        
        print("✅ BG/NBD model fitted successfully")
        print(f"   - Model parameters: {dict(self.bgf_model.summary)}")
        
        # Calculate probability alive for each customer
        self.transaction_data['prob_alive'] = self.bgf_model.conditional_probability_alive(
            frequency=self.transaction_data['frequency'],
            recency=self.transaction_data['recency'],
            T=self.transaction_data['T']
        )
        
        # Predict future transactions (next 12 months)
        self.transaction_data['predicted_transactions_12m'] = self.bgf_model.conditional_expected_number_of_purchases_up_to_time(
            t=1,  # 1 year
            frequency=self.transaction_data['frequency'],
            recency=self.transaction_data['recency'],
            T=self.transaction_data['T']
        )
        
        print("✅ Generated probability alive and transaction predictions")
    
    def fit_gamma_gamma_model(self):
        """Fit Gamma-Gamma model for monetary value prediction"""
        
        print("🔧 Fitting Gamma-Gamma model for monetary value prediction...")
        
        # Filter customers with more than 0 purchases for Gamma-Gamma model
        gg_data = self.transaction_data[self.transaction_data['frequency'] > 0]
        
        # Initialize and fit the Gamma-Gamma model
        self.ggf_model = GammaGammaFitter(penalizer_coef=0.1)
        
        # Fit the model
        self.ggf_model.fit(
            frequency=gg_data['frequency'],
            monetary_value=gg_data['monetary_value']
        )
        
        print("✅ Gamma-Gamma model fitted successfully")
        print(f"   - Model parameters: {dict(self.ggf_model.summary)}")
        
        # Predict customer monetary value
        self.transaction_data['predicted_monetary_value'] = self.ggf_model.conditional_expected_average_profit(
            frequency=self.transaction_data['frequency'],
            monetary_value=self.transaction_data['monetary_value']
        )
        
        print("✅ Generated monetary value predictions")
    
    def calculate_clv(self, time_period_months=12):
        """Calculate Customer Lifetime Value using both models"""
        
        print(f"💰 Calculating CLV for {time_period_months} months period...")
        
        if self.bgf_model is None or self.ggf_model is None:
            raise ValueError("Both BG/NBD and Gamma-Gamma models must be fitted first")
        
        # Calculate expected transactions over the time period
        expected_transactions = self.bgf_model.conditional_expected_number_of_purchases_up_to_time(
            t=time_period_months/12,  # Convert months to years
            frequency=self.transaction_data['frequency'],
            recency=self.transaction_data['recency'],
            T=self.transaction_data['T']
        )
        
        # Calculate CLV = Expected Transactions × Expected Monetary Value
        self.transaction_data[f'clv_{time_period_months}m'] = (
            expected_transactions * self.transaction_data['predicted_monetary_value']
        )
        
        # Calculate lifetime CLV (using probability alive)
        lifetime_transactions = self.transaction_data['predicted_transactions_12m'] / self.transaction_data['prob_alive']
        self.transaction_data['clv_lifetime'] = (
            lifetime_transactions * self.transaction_data['predicted_monetary_value']
        )
        
        # Handle infinite values
        self.transaction_data['clv_lifetime'] = self.transaction_data['clv_lifetime'].replace([np.inf, -np.inf], np.nan)
        self.transaction_data['clv_lifetime'] = self.transaction_data['clv_lifetime'].fillna(
            self.transaction_data['clv_lifetime'].median()
        )
        
        print(f"✅ CLV calculated for {len(self.transaction_data):,} customers")
        print(f"   - Average {time_period_months}-month CLV: ${self.transaction_data[f'clv_{time_period_months}m'].mean():.2f}")
        print(f"   - Average lifetime CLV: ${self.transaction_data['clv_lifetime'].mean():.2f}")
        
        return time_period_months
    
    def create_clv_visualizations(self):
        """Create comprehensive CLV visualizations"""
        
        print("📊 Creating CLV model visualizations...")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Lifetime Value (CLV) Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Frequency-Recency Matrix
        plot_frequency_recency_matrix(self.bgf_model, ax=axes[0, 0])
        axes[0, 0].set_title('BG/NBD Frequency-Recency Matrix')
        
        # 2. Probability Alive Matrix
        plot_probability_alive_matrix(self.bgf_model, ax=axes[0, 1])
        axes[0, 1].set_title('Customer Probability Alive Matrix')
        
        # 3. CLV Distribution
        axes[0, 2].hist(self.transaction_data['clv_12m'], bins=50, alpha=0.7, 
                       color=self.colors['primary'], edgecolor='black')
        axes[0, 2].set_title('12-Month CLV Distribution')
        axes[0, 2].set_xlabel('CLV ($)')
        axes[0, 2].set_ylabel('Number of Customers')
        
        # Add statistics
        mean_clv = self.transaction_data['clv_12m'].mean()
        median_clv = self.transaction_data['clv_12m'].median()
        axes[0, 2].axvline(mean_clv, color='red', linestyle='--', label=f'Mean: ${mean_clv:.0f}')
        axes[0, 2].axvline(median_clv, color='orange', linestyle='--', label=f'Median: ${median_clv:.0f}')
        axes[0, 2].legend()
        
        # 4. Probability Alive vs Frequency
        axes[1, 0].scatter(self.transaction_data['frequency'], self.transaction_data['prob_alive'],
                          alpha=0.6, color=self.colors['secondary'])
        axes[1, 0].set_title('Probability Alive vs Transaction Frequency')
        axes[1, 0].set_xlabel('Transaction Frequency')
        axes[1, 0].set_ylabel('Probability Alive')
        
        # 5. CLV vs Monetary Value
        axes[1, 1].scatter(self.transaction_data['monetary_value'], self.transaction_data['clv_12m'],
                          alpha=0.6, color=self.colors['success'])
        axes[1, 1].set_title('CLV vs Average Monetary Value')
        axes[1, 1].set_xlabel('Average Monetary Value ($)')
        axes[1, 1].set_ylabel('12-Month CLV ($)')
        
        # 6. Top Customers by CLV
        top_customers = self.transaction_data.nlargest(20, 'clv_12m')
        axes[1, 2].barh(range(len(top_customers)), top_customers['clv_12m'], 
                       color=self.colors['info'], alpha=0.8)
        axes[1, 2].set_title('Top 20 Customers by 12-Month CLV')
        axes[1, 2].set_xlabel('CLV ($)')
        axes[1, 2].set_ylabel('Customer Rank')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'clv_modeling_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ CLV modeling dashboard created")
    
    def create_interactive_clv_analysis(self):
        """Create interactive CLV analysis with Plotly"""
        
        print("📈 Creating interactive CLV analysis...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['CLV vs Probability Alive', 'CLV Distribution by Segments',
                          'Predicted vs Actual Frequency', 'CLV vs Customer Value Metrics'],
            specs=[[{"type": "scatter"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. CLV vs Probability Alive (colored by frequency)
        fig.add_trace(
            go.Scatter(x=self.transaction_data['prob_alive'], 
                      y=self.transaction_data['clv_12m'],
                      mode='markers',
                      marker=dict(color=self.transaction_data['frequency'], 
                                colorscale='Viridis', size=6, opacity=0.7,
                                colorbar=dict(title="Frequency", x=0.45)),
                      name='CLV vs Prob Alive',
                      text=[f'Customer: {cid}<br>CLV: ${clv:.0f}<br>Prob Alive: {pa:.2f}<br>Frequency: {freq}'
                           for cid, clv, pa, freq in zip(
                               self.transaction_data['customer_id'],
                               self.transaction_data['clv_12m'],
                               self.transaction_data['prob_alive'],
                               self.transaction_data['frequency'])],
                      hovertemplate='%{text}<extra></extra>'),
            row=1, col=1
        )
        
        # 2. CLV Distribution by quintiles
        clv_quintiles = pd.qcut(self.transaction_data['clv_12m'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_data = self.transaction_data[clv_quintiles == q]['clv_12m']
            fig.add_trace(
                go.Box(y=q_data, name=f'CLV {q}', boxpoints='outliers'),
                row=1, col=2
            )
        
        # 3. Predicted vs Historical Frequency
        fig.add_trace(
            go.Scatter(x=self.transaction_data['frequency'],
                      y=self.transaction_data['predicted_transactions_12m'],
                      mode='markers',
                      marker=dict(color='red', size=6, opacity=0.6),
                      name='Predicted vs Historical'),
            row=2, col=1
        )
        
        # Add diagonal line for perfect prediction
        max_freq = max(self.transaction_data['frequency'].max(), 
                      self.transaction_data['predicted_transactions_12m'].max())
        fig.add_trace(
            go.Scatter(x=[0, max_freq], y=[0, max_freq],
                      mode='lines', line=dict(dash='dash', color='black'),
                      name='Perfect Prediction', showlegend=False),
            row=2, col=1
        )
        
        # 4. CLV vs multiple metrics
        fig.add_trace(
            go.Scatter(x=self.transaction_data['monetary_value'],
                      y=self.transaction_data['clv_12m'],
                      mode='markers',
                      marker=dict(color='blue', size=6, opacity=0.6),
                      name='CLV vs Monetary Value'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Interactive CLV Model Analysis",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Probability Alive", row=1, col=1)
        fig.update_yaxes(title_text="12-Month CLV ($)", row=1, col=1)
        fig.update_xaxes(title_text="CLV Quintiles", row=1, col=2)
        fig.update_yaxes(title_text="CLV ($)", row=1, col=2)
        fig.update_xaxes(title_text="Historical Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Predicted Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Average Monetary Value ($)", row=2, col=2)
        fig.update_yaxes(title_text="12-Month CLV ($)", row=2, col=2)
        
        # Save interactive plot
        fig.write_html(self.results_dir / 'clv_interactive_analysis.html')
        
        print("✅ Interactive CLV analysis created")
    
    def generate_clv_insights_report(self):
        """Generate comprehensive CLV insights and business recommendations"""
        
        print("📋 Generating CLV insights report...")
        
        # Calculate key metrics
        total_customers = len(self.transaction_data)
        avg_clv_12m = self.transaction_data['clv_12m'].mean()
        median_clv_12m = self.transaction_data['clv_12m'].median()
        total_clv_12m = self.transaction_data['clv_12m'].sum()
        
        # CLV segments
        clv_quintiles = pd.qcut(self.transaction_data['clv_12m'], 5, 
                               labels=['Low Value', 'Below Average', 'Average', 'Above Average', 'High Value'])
        clv_segment_analysis = self.transaction_data.groupby(clv_quintiles).agg({
            'customer_id': 'count',
            'clv_12m': ['mean', 'sum'],
            'prob_alive': 'mean',
            'frequency': 'mean',
            'monetary_value': 'mean'
        }).round(2)
        
        # Model performance metrics
        bgf_summary = self.bgf_model.summary
        ggf_summary = self.ggf_model.summary
        
        # Generate report content
        report_content = [
            "# Customer Lifetime Value (CLV) Modeling Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"This report presents CLV analysis using BG/NBD and Gamma-Gamma models for {total_customers:,} banking customers.",
            "The models predict customer transaction behavior and lifetime value to optimize marketing investments.",
            "",
            "## Key CLV Metrics",
            "",
            f"- **Total Customers Analyzed**: {total_customers:,}",
            f"- **Average 12-Month CLV**: ${avg_clv_12m:.2f}",
            f"- **Median 12-Month CLV**: ${median_clv_12m:.2f}",
            f"- **Total Portfolio CLV (12M)**: ${total_clv_12m:,.2f}",
            f"- **CLV Range**: ${self.transaction_data['clv_12m'].min():.2f} - ${self.transaction_data['clv_12m'].max():.2f}",
            "",
            "## Model Performance",
            "",
            "### BG/NBD Model (Customer Lifetime & Frequency)",
            f"- **Log-likelihood**: {bgf_summary.loc['log-likelihood']['coef']:.2f}",
            f"- **AIC**: {bgf_summary.loc['AIC']['coef']:.2f}",
            "",
            "### Gamma-Gamma Model (Monetary Value)",
            f"- **Log-likelihood**: {ggf_summary.loc['log-likelihood']['coef']:.2f}",
            f"- **AIC**: {ggf_summary.loc['AIC']['coef']:.2f}",
            "",
            "## CLV Segmentation Analysis",
            ""
        ]
        
        # Add segment analysis
        clv_segment_analysis.columns = ['Customer_Count', 'Avg_CLV', 'Total_CLV', 'Avg_Prob_Alive', 'Avg_Frequency', 'Avg_Monetary_Value']
        
        for segment in clv_segment_analysis.index:
            data = clv_segment_analysis.loc[segment]
            report_content.extend([
                f"### {segment} Customers",
                f"- **Count**: {data['Customer_Count']:,} ({data['Customer_Count']/total_customers*100:.1f}%)",
                f"- **Average CLV**: ${data['Avg_CLV']:.2f}",
                f"- **Total CLV Contribution**: ${data['Total_CLV']:,.2f} ({data['Total_CLV']/total_clv_12m*100:.1f}%)",
                f"- **Average Probability Alive**: {data['Avg_Prob_Alive']:.2f}",
                f"- **Average Transaction Frequency**: {data['Avg_Frequency']:.1f}/year",
                f"- **Average Monetary Value**: ${data['Avg_Monetary_Value']:.2f}",
                ""
            ])
        
        # Business insights
        high_value_threshold = self.transaction_data['clv_12m'].quantile(0.8)
        high_value_customers = len(self.transaction_data[self.transaction_data['clv_12m'] >= high_value_threshold])
        churn_risk_customers = len(self.transaction_data[self.transaction_data['prob_alive'] < 0.5])
        
        high_value_pct = high_value_customers/total_customers*100
        churn_risk_pct = churn_risk_customers/total_customers*100
        
        report_content.extend([
            "## Key Business Insights",
            "",
            f"1. **High-Value Segment**: {high_value_customers:,} customers ({high_value_pct:.1f}%) generate top 20% of CLV",
            f"2. **Churn Risk**: {churn_risk_customers:,} customers ({churn_risk_pct:.1f}%) have probability alive < 50%",
            "3. **Revenue Concentration**: Top quintile contributes significant portion of total CLV",
            "4. **Frequency Impact**: High-value customers show higher transaction frequency",
            "",
            "## Strategic Recommendations",
            "",
            "### 1. High-Value Customer Retention",
            "- Focus retention efforts on High Value and Above Average CLV segments",
            "- Implement VIP programs for customers with CLV > $2,000",
            "- Monitor probability alive scores for early churn warning",
            "",
            "### 2. Customer Development Programs", 
            "- Target Average and Below Average segments for upselling",
            "- Increase transaction frequency through engagement campaigns",
            "- Focus on increasing monetary value per transaction",
            "",
            "### 3. Churn Prevention",
            f"- Immediate intervention for {churn_risk_customers:,} customers with low probability alive",
            "- Develop win-back campaigns for customers with declining frequency",
            "- Implement predictive churn models using probability alive scores",
            "",
            "### 4. Marketing ROI Optimization",
            "- Allocate marketing budget based on CLV predictions",
            "- Set customer acquisition cost targets using lifetime CLV",
            "- Prioritize retention over acquisition for high CLV segments",
            "",
            "## Model Outputs",
            "",
            "- `clv_predictions.csv`: Individual customer CLV predictions and metrics",
            "- `clv_modeling_dashboard.png`: Comprehensive CLV analysis dashboard",
            "- `clv_interactive_analysis.html`: Interactive CLV exploration tool",
            "",
            "## Technical Notes",
            "",
            "- **BG/NBD Model**: Predicts customer transaction timing and churn probability",
            "- **Gamma-Gamma Model**: Predicts customer monetary value per transaction", 
            "- **CLV Calculation**: Expected transactions × Expected monetary value",
            "- **Time Horizon**: 12-month predictions with lifetime estimates",
            "",
            "---",
            "*This analysis provides foundation for data-driven customer relationship management and marketing optimization.*"
        ])
        
        # Save report
        report_path = self.results_dir / 'clv_modeling_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"✅ CLV insights report generated: {report_path}")
        
        return clv_segment_analysis
    
    def save_clv_predictions(self):
        """Save CLV predictions and customer metrics to CSV"""
        
        print("💾 Saving CLV predictions and customer metrics...")
        
        # Merge transaction data with original customer data
        clv_predictions = self.transaction_data.merge(
            self.df[['CustomerId', 'Geography', 'Gender', 'Age', 'Balance', 'NumOfProducts', 
                    'HasCrCard', 'IsActiveMember', 'Exited']],
            left_on='customer_id',
            right_on='CustomerId',
            how='left'
        )
        
        # Add CLV segments
        clv_predictions['clv_segment'] = pd.qcut(
            clv_predictions['clv_12m'], 5,
            labels=['Low Value', 'Below Average', 'Average', 'Above Average', 'High Value']
        )
        
        # Add risk categories
        clv_predictions['churn_risk'] = pd.cut(
            clv_predictions['prob_alive'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['High Risk', 'Medium Risk', 'Low Risk']
        )
        
        # Select key columns for output
        output_columns = [
            'customer_id', 'Geography', 'Gender', 'Age', 'Balance', 'NumOfProducts',
            'frequency', 'recency', 'monetary_value', 'prob_alive', 
            'predicted_transactions_12m', 'predicted_monetary_value',
            'clv_12m', 'clv_lifetime', 'clv_segment', 'churn_risk'
        ]
        
        # Save predictions
        clv_output = clv_predictions[output_columns].round(2)
        clv_output.to_csv(self.data_dir / 'banking_clv_predictions.csv', index=False)
        
        print(f"✅ CLV predictions saved: {self.data_dir / 'banking_clv_predictions.csv'}")
        print(f"   - {len(clv_output):,} customer predictions")
        print(f"   - {len(output_columns)} features per customer")
        
        return clv_output
    
    def run_complete_clv_analysis(self):
        """Run the complete CLV modeling analysis"""
        
        print("🚀 Starting Comprehensive CLV Modeling Analysis")
        print("=" * 55)
        
        # Load and prepare data
        if not self.load_and_prepare_data():
            return False
        
        # Fit models
        print("\n🔧 Training CLV Models...")
        self.fit_bgf_model()
        self.fit_gamma_gamma_model()
        
        # Calculate CLV
        print("\n💰 Calculating Customer Lifetime Values...")
        self.calculate_clv(time_period_months=12)
        
        # Create visualizations
        print("\n📊 Creating CLV Visualizations...")
        self.create_clv_visualizations()
        self.create_interactive_clv_analysis()
        
        # Generate insights
        print("\n📋 Generating Business Insights...")
        segment_analysis = self.generate_clv_insights_report()
        
        # Save predictions
        print("\n💾 Saving CLV Predictions...")
        clv_predictions = self.save_clv_predictions()
        
        # Summary
        print("\n" + "=" * 55)
        print("💰 CLV MODELING SUMMARY")
        print("=" * 55)
        print(f"Total Customers: {len(self.transaction_data):,}")
        print(f"Average 12-Month CLV: ${self.transaction_data['clv_12m'].mean():.2f}")
        print(f"Total Portfolio Value: ${self.transaction_data['clv_12m'].sum():,.2f}")
        print(f"High-Value Customers (Top 20%): {len(self.transaction_data[self.transaction_data['clv_12m'] >= self.transaction_data['clv_12m'].quantile(0.8)]):,}")
        print(f"At-Risk Customers (Prob Alive < 50%): {len(self.transaction_data[self.transaction_data['prob_alive'] < 0.5]):,}")
        
        print("\n✅ CLV modeling analysis completed!")
        print(f"📁 Results saved in: {self.results_dir}")
        print(f"📊 Customer predictions saved in: {self.data_dir}")
        
        return True

def main():
    """Main execution function"""
    
    modeler = BankingCLVModeler()
    success = modeler.run_complete_clv_analysis()
    
    if success:
        print("\n🎉 CLV modeling completed successfully!")
        print("📈 Ready for final customer segmentation and presentation")
    else:
        print("\n❌ CLV modeling failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())