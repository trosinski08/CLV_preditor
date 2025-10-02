"""
Banking Dataset Explorer for CLV Analysis
Explores the banking customer dataset and prepares it for CLV modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_banking_dataset():
    """
    Comprehensive exploration of the banking dataset for CLV analysis
    """
    print("="*60)
    print("BANKING DATASET EXPLORATION FOR CLV ANALYSIS")
    print("="*60)
    
    # Load the dataset
    try:
        df = pd.read_csv('data/banking_clv_dataset.csv')
        print(f"✅ Banking dataset loaded successfully!")
    except FileNotFoundError:
        print("❌ Banking dataset not found. Please run load_banking_clv_data.py first.")
        return None
    
    print(f"\n📊 BASIC DATASET INFORMATION")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Dataset columns analysis
    print(f"\n📋 COLUMN ANALYSIS")
    print("Columns and their types:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        print(f"  {col:20} | {str(dtype):10} | Nulls: {null_count:5} | Unique: {unique_count:6}")
    
    # Customer segments analysis
    print(f"\n🏛️ BANKING CUSTOMER ANALYSIS")
    
    # Geography distribution
    if 'Geography' in df.columns:
        print(f"\n🌍 Geographic Distribution:")
        geo_dist = df['Geography'].value_counts()
        for country, count in geo_dist.items():
            print(f"  {country}: {count:,} customers ({count/len(df)*100:.1f}%)")
    
    # Age analysis
    if 'Age' in df.columns:
        print(f"\n👥 Age Statistics:")
        print(f"  Average age: {df['Age'].mean():.1f} years")
        print(f"  Age range: {df['Age'].min()} - {df['Age'].max()} years")
        age_groups = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
        print(f"  Age distribution:")
        for group, count in age_groups.value_counts().items():
            print(f"    {group}: {count:,} customers ({count/len(df)*100:.1f}%)")
    
    # Account balance analysis
    if 'Balance' in df.columns:
        print(f"\n💰 ACCOUNT BALANCE ANALYSIS")
        print(f"  Balance statistics:")
        balance_stats = df['Balance'].describe()
        for stat, value in balance_stats.items():
            print(f"    {stat}: ${value:,.2f}")
        
        # Balance segments
        df['BalanceSegment'] = pd.cut(df['Balance'], 
                                     bins=[0, 1, 50000, 100000, 200000, float('inf')],
                                     labels=['Zero', 'Low (<50K)', 'Medium (50-100K)', 'High (100-200K)', 'Very High (>200K)'])
        print(f"\n  Balance segments:")
        for segment, count in df['BalanceSegment'].value_counts().items():
            avg_balance = df[df['BalanceSegment'] == segment]['Balance'].mean()
            print(f"    {segment}: {count:,} customers (${avg_balance:,.0f} avg)")
    
    # Product usage analysis
    if 'NumOfProducts' in df.columns:
        print(f"\n🏦 PRODUCT USAGE ANALYSIS")
        product_dist = df['NumOfProducts'].value_counts().sort_index()
        print(f"  Products per customer:")
        for products, count in product_dist.items():
            print(f"    {products} product(s): {count:,} customers ({count/len(df)*100:.1f}%)")
        
        # Cross-selling opportunity
        avg_products = df['NumOfProducts'].mean()
        print(f"  Average products per customer: {avg_products:.2f}")
    
    # Customer loyalty analysis
    if 'Tenure' in df.columns:
        print(f"\n⏰ CUSTOMER TENURE ANALYSIS")
        print(f"  Tenure statistics (years with bank):")
        tenure_stats = df['Tenure'].describe()
        for stat, value in tenure_stats.items():
            print(f"    {stat}: {value:.1f} years")
        
        # Tenure segments
        df['TenureSegment'] = pd.cut(df['Tenure'], 
                                    bins=[0, 2, 5, 8, float('inf')],
                                    labels=['New (0-2y)', 'Growing (2-5y)', 'Mature (5-8y)', 'Loyal (8y+)'])
        print(f"\\n  Tenure segments:")
        for segment, count in df['TenureSegment'].value_counts().items():
            print(f"    {segment}: {count:,} customers ({count/len(df)*100:.1f}%)")
    
    # Customer activity analysis
    if 'IsActiveMember' in df.columns:
        active_customers = df['IsActiveMember'].sum()
        print(f"\n🔥 CUSTOMER ACTIVITY")
        print(f"  Active customers: {active_customers:,} ({active_customers/len(df)*100:.1f}%)")
        print(f"  Inactive customers: {len(df)-active_customers:,} ({(len(df)-active_customers)/len(df)*100:.1f}%)")
    
    # Churn analysis
    if 'Exited' in df.columns:
        churned_customers = df['Exited'].sum()
        print(f"\n📉 CHURN ANALYSIS")
        print(f"  Churned customers: {churned_customers:,} ({churned_customers/len(df)*100:.1f}%)")
        print(f"  Retained customers: {len(df)-churned_customers:,} ({(len(df)-churned_customers)/len(df)*100:.1f}%)")
        
        # Churn by segment
        if 'BalanceSegment' in df.columns:
            churn_by_balance = df.groupby('BalanceSegment')['Exited'].agg(['count', 'sum', 'mean']).round(3)
            churn_by_balance.columns = ['Total', 'Churned', 'ChurnRate']
            print(f"\\n  Churn rate by balance segment:")
            print(churn_by_balance)
    
    # Credit worthiness
    if 'CreditScore' in df.columns:
        print(f"\n💳 CREDIT SCORE ANALYSIS")
        credit_stats = df['CreditScore'].describe()
        print(f"  Credit score statistics:")
        for stat, value in credit_stats.items():
            print(f"    {stat}: {value:.0f}")
        
        # Credit segments
        df['CreditSegment'] = pd.cut(df['CreditScore'], 
                                    bins=[0, 580, 670, 740, 800, 850],
                                    labels=['Poor (<580)', 'Fair (580-670)', 'Good (670-740)', 'Very Good (740-800)', 'Excellent (800+)'])
        print(f"\\n  Credit score segments:")
        for segment, count in df['CreditSegment'].value_counts().items():
            avg_balance = df[df['CreditSegment'] == segment]['Balance'].mean()
            print(f"    {segment}: {count:,} customers (${avg_balance:,.0f} avg balance)")
    
    # Income analysis
    if 'EstimatedSalary' in df.columns:
        print(f"\n💼 INCOME ANALYSIS")
        income_stats = df['EstimatedSalary'].describe()
        print(f"  Estimated salary statistics:")
        for stat, value in income_stats.items():
            print(f"    {stat}: ${value:,.2f}")
        
        # Income vs Balance correlation
        if 'Balance' in df.columns:
            correlation = df['EstimatedSalary'].corr(df['Balance'])
            print(f"  Correlation between salary and balance: {correlation:.3f}")
    
    # Prepare CLV components
    print(f"\n🎯 CLV COMPONENTS PREPARATION")
    
    # Create monetary value (using balance as proxy for customer value)
    if 'Balance' in df.columns:
        df['MonetaryValue'] = df['Balance']
        print(f"  ✅ Monetary component: Account Balance")
    
    # Create frequency (using number of products as proxy)
    if 'NumOfProducts' in df.columns:
        df['Frequency'] = df['NumOfProducts']
        print(f"  ✅ Frequency component: Number of Products")
    
    # Create recency (using tenure inversely - longer tenure = more recent engagement)
    if 'Tenure' in df.columns:
        df['Recency'] = df['Tenure'].max() - df['Tenure'] + 1  # Invert so newer customers have lower recency
        print(f"  ✅ Recency component: Inverted Tenure")
    
    # Customer Lifetime Value estimation
    print(f"\n💰 PRELIMINARY CLV ESTIMATION")
    
    if all(col in df.columns for col in ['MonetaryValue', 'Frequency', 'Tenure', 'IsActiveMember']):
        # Simple CLV calculation: (Average Annual Value) * (Expected Lifetime) * (Retention Rate)
        
        # Estimate annual value based on balance and products
        df['EstimatedAnnualValue'] = (df['MonetaryValue'] * 0.02) + (df['Frequency'] * 1000)  # 2% on balance + 1000 per product
        
        # Estimate lifetime based on tenure and activity
        df['EstimatedLifetime'] = df['Tenure'] + (df['IsActiveMember'] * 5)  # Current tenure + 5 years if active
        
        # Simple CLV calculation
        df['EstimatedCLV'] = df['EstimatedAnnualValue'] * df['EstimatedLifetime'] * (1 - df['Exited'])
        
        print(f"  CLV Statistics:")
        clv_stats = df['EstimatedCLV'].describe()
        for stat, value in clv_stats.items():
            print(f"    {stat}: ${value:,.2f}")
        
        # Top customers by CLV
        top_customers = df.nlargest(10, 'EstimatedCLV')[['CustomerId', 'EstimatedCLV', 'Balance', 'NumOfProducts', 'Tenure', 'Geography']]
        print(f"\\n  Top 10 customers by estimated CLV:")
        print(top_customers.to_string(index=False))
    
    # Sample data for verification
    print(f"\\n📋 SAMPLE DATA")
    print(df.head(10).to_string())
    
    # Save enhanced dataset
    output_file = 'data/banking_clv_enhanced.csv'
    df.to_csv(output_file, index=False)
    print(f"\\n💾 Enhanced dataset saved: {output_file}")
    
    return df

if __name__ == "__main__":
    df = explore_banking_dataset()