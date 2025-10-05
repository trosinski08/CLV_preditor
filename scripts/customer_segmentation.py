"""
Customer Segmentation Script
===========================

Creates final actionable customer segments by combining:
- Predicted CLV (12m and lifetime)
- RFM Segments
- Churn Risk (probability alive proxy)
- Revenue Tier (from revenue modeling)

Outputs:
- data/final_customer_segments.csv
- results/final_segmentation_report.md
- results/final_segmentation_dashboard.png
- results/customer_value_matrix.html

Author: Banking CLV Analysis Team
Date: 2025-10-05
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentationBuilder:
    def __init__(self,
                 clv_path='data/banking_clv_predictions.csv',
                 rfm_path='data/banking_rfm_analysis.csv',
                 cleaned_path='data/banking_clv_cleaned.csv'):
        self.clv_path = Path(clv_path)
        self.rfm_path = Path(rfm_path)
        self.cleaned_path = Path(cleaned_path)
        self.data_dir = Path('data')
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        self.df = None

    def load_data(self):
        print('📥 Loading source datasets...')
        clv = pd.read_csv(self.clv_path)
        rfm = pd.read_csv(self.rfm_path)
        cleaned = pd.read_csv(self.cleaned_path)
        # Normalize key for merge
        if 'CustomerId' in cleaned.columns:
            cleaned_key = 'CustomerId'
        else:
            cleaned_key = 'customer_id'
        # Merge progressively
        merged = clv.merge(rfm, left_on='customer_id', right_on='CustomerId', how='left', suffixes=('', '_rfm'))
        # Select columns safely (EstimatedSalary may or may not exist)
        extra_cols = [col for col in ['EstimatedSalary'] if col in cleaned.columns]
        if 'revenue_tier' in cleaned.columns:
            extra_cols.append('revenue_tier')
        sel_cols = [cleaned_key] + extra_cols
        merged = merged.merge(cleaned[sel_cols], left_on='customer_id', right_on=cleaned_key, how='left')
        # If revenue_tier missing, derive from Balance quantiles (available via clv merge -> later from cleaned if needed)
        if 'revenue_tier' not in merged.columns:
            if 'Balance' in merged.columns:
                q = merged['Balance'].quantile([0.25,0.5,0.75,0.9])
                def rt(v):
                    if v <= q[0.25]: return 'Low Value'
                    if v <= q[0.50]: return 'Medium Value'
                    if v <= q[0.75]: return 'High Value'
                    if v <= q[0.90]: return 'Premium'
                    return 'VIP'
                merged['revenue_tier'] = merged['Balance'].apply(rt)
            else:
                merged['revenue_tier'] = 'Unknown'
        self.df = merged
        print(f'✅ Merged dataset shape: {self.df.shape}')

    def create_clv_tiers(self):
        print('🔧 Creating CLV tiers...')
        # Use quantiles for High/Medium/Low
        q_low = self.df['clv_12m'].quantile(0.33)
        q_high = self.df['clv_12m'].quantile(0.66)
        def clv_bucket(v):
            if v <= q_low:
                return 'Low'
            if v <= q_high:
                return 'Medium'
            return 'High'
        self.df['CLV_Tier'] = self.df['clv_12m'].apply(clv_bucket)
        # Lifetime tier
        lt_low = self.df['clv_lifetime'].quantile(0.33)
        lt_high = self.df['clv_lifetime'].quantile(0.66)
        def clv_life_bucket(v):
            if v <= lt_low:
                return 'Low'
            if v <= lt_high:
                return 'Medium'
            return 'High'
        self.df['CLV_Lifetime_Tier'] = self.df['clv_lifetime'].apply(clv_life_bucket)
        print('✅ CLV tiers added')

    def map_churn_risk_numeric(self):
        print('🔧 Mapping churn risk levels...')
        risk_order = {'High Risk': 3, 'Medium Risk': 2, 'Low Risk': 1}
        if 'churn_risk' not in self.df.columns:
            # Derive from prob_alive
            self.df['churn_risk'] = pd.cut(self.df['prob_alive'], bins=[0,0.3,0.7,1.0], labels=['High Risk','Medium Risk','Low Risk'])
        self.df['churn_risk_score'] = self.df['churn_risk'].map(risk_order).fillna(2)
        print('✅ Churn risk mapped')

    def create_actionable_segment(self):
        print('🧭 Building actionable segment labels...')
        def label_row(r):
            # Priority to churn risk + value
            if r['CLV_Tier']=='High' and r['churn_risk']=='Low Risk':
                return 'Strategic Champion'
            if r['CLV_Tier']=='High' and r['churn_risk']=='High Risk':
                return 'Save High-Value'
            if r['CLV_Tier']=='Medium' and r['churn_risk']=='High Risk':
                return 'Rescue Medium'
            if r['CLV_Tier']=='Low' and r['churn_risk']=='High Risk':
                return 'Low Value Churn'
            if r['CLV_Tier']=='High' and r['churn_risk']=='Medium Risk':
                return 'Nurture High'
            if r['CLV_Tier']=='Medium' and r['churn_risk']=='Low Risk':
                return 'Growth Stable'
            if r['CLV_Tier']=='Low' and r['churn_risk']=='Low Risk':
                return 'Low Maintenance'
            return 'Monitor'
        self.df['Action_Segment'] = self.df.apply(label_row, axis=1)
        print('✅ Actionable segments created')

    def compute_segment_kpis(self):
        print('📊 Computing segment KPIs...')
        group_cols = ['Action_Segment']
        kpis = self.df.groupby(group_cols).agg({
            'customer_id':'count',
            'clv_12m':'mean',
            'clv_lifetime':'mean',
            'prob_alive':'mean',
            'frequency':'mean',
            'monetary_value':'mean'
        }).rename(columns={'customer_id':'Customers','clv_12m':'Avg_CLV_12m','clv_lifetime':'Avg_CLV_Lifetime','prob_alive':'Avg_Prob_Alive','frequency':'Avg_Frequency','monetary_value':'Avg_Monetary'}).round(2)
        kpis['Share_%'] = (kpis['Customers']/len(self.df)*100).round(2)
        self.segment_kpis = kpis.sort_values('Avg_CLV_12m', ascending=False)
        print('✅ KPIs computed')

    def save_outputs(self):
        print('💾 Saving outputs...')
        out_path = self.data_dir / 'final_customer_segments.csv'
        self.df.to_csv(out_path, index=False)
        print(f'✅ Saved detailed segments: {out_path}')
        # Save KPIs
        self.segment_kpis.to_csv(self.data_dir / 'segment_kpis_summary.csv')

    def plot_dashboard(self):
        print('📈 Creating segmentation dashboard...')
        fig, axes = plt.subplots(2,2, figsize=(16,12))
        fig.suptitle('Final Customer Segmentation Dashboard', fontsize=16, fontweight='bold')
        # Distribution of actionable segments
        self.df['Action_Segment'].value_counts().plot(kind='bar', ax=axes[0,0], color='#1f77b4')
        axes[0,0].set_title('Action Segment Counts')
        axes[0,0].set_ylabel('Customers')
        # CLV by segment
        sns.barplot(x=self.segment_kpis.index, y='Avg_CLV_12m', data=self.segment_kpis, ax=axes[0,1], palette='viridis')
        axes[0,1].set_title('Avg 12M CLV by Segment')
        axes[0,1].tick_params(axis='x', rotation=45)
        # Probability alive vs CLV scatter
        axes[1,0].scatter(self.df['prob_alive'], self.df['clv_12m'], alpha=0.4, s=10)
        axes[1,0].set_title('CLV vs Probability Alive')
        axes[1,0].set_xlabel('Probability Alive')
        axes[1,0].set_ylabel('12M CLV')
        # Segment share pie
        self.segment_kpis['Customers'].plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%', startangle=90)
        axes[1,1].set_ylabel('')
        axes[1,1].set_title('Customer Share by Segment')
        plt.tight_layout()
        dash_path = self.results_dir / 'final_segmentation_dashboard.png'
        plt.savefig(dash_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'✅ Dashboard saved: {dash_path}')

    def interactive_matrix(self):
        print('🧪 Creating interactive customer value matrix...')
        fig = px.scatter(self.df, x='prob_alive', y='clv_12m', color='Action_Segment',
                         size='monetary_value', hover_data=['frequency','CLV_Tier','churn_risk'],
                         title='Customer Value Matrix (Probability Alive vs 12M CLV)', opacity=0.6)
        html_path = self.results_dir / 'customer_value_matrix.html'
        fig.write_html(html_path)
        print(f'✅ Interactive matrix saved: {html_path}')

    def write_report(self):
        print('📝 Writing segmentation report...')
        lines = [
            '# Final Customer Segmentation Report',
            '',
            f'Total customers analyzed: {len(self.df):,}',
            '',
            '## Segment KPIs',
            '',
        ]
        # Attempt markdown table; fallback to simple text if dependency missing
        try:
            lines.append(self.segment_kpis.to_markdown())
        except Exception:
            lines.append('Segment KPIs (plain text, install tabulate for markdown table)')
            lines.append(self.segment_kpis.to_csv())
        lines += [
            '',
            '## Key Strategic Recommendations',
            '',
            '- Focus retention on Strategic Champion & Save High-Value segments',
            '- Launch targeted win-back for Save High-Value and Rescue Medium',
            '- Automate nurture flows for Growth Stable segment',
            '- Optimize cost-to-serve for Low Maintenance cohort',
            '',
            '## Next Steps',
            '',
            '1. Integrate segments into CRM',
            '2. Align marketing budget with segment CLV',
            '3. Add predictive churn uplift modeling',
            '4. Monitor segment migration monthly',
            '',
            '---',
            '*Generated automatically by customer_segmentation.py*'
        ]
        report_path = self.results_dir / 'final_segmentation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'✅ Report saved: {report_path}')

    def run(self):
        self.load_data()
        self.create_clv_tiers()
        self.map_churn_risk_numeric()
        self.create_actionable_segment()
        self.compute_segment_kpis()
        self.save_outputs()
        # Try catch for visualization errors
        try:
            self.plot_dashboard()
        except Exception as e:
            print(f'⚠️ Dashboard generation failed: {e}')
        try:
            self.interactive_matrix()
        except Exception as e:
            print(f'⚠️ Interactive matrix failed: {e}')
        self.write_report()
        print('🎉 Segmentation pipeline complete')

def main():
    builder = CustomerSegmentationBuilder()
    builder.run()

if __name__ == '__main__':
    main()
