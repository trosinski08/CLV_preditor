"""
Improved Banking Data Loader for CLV Project
Focuses on datasets suitable for Customer Lifetime Value analysis
"""

import kagglehub
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import requests

def search_clv_suitable_datasets():
    """
    Search for datasets specifically suitable for CLV analysis in banking context
    """
    print("="*60)
    print("SEARCHING FOR CLV-SUITABLE BANKING DATASETS")
    print("="*60)
    
    # Datasets focused on customer behavior, transactions, and lifetime value
    clv_datasets = [
        # Banking customer behavior
        ("shivamb/bank-customer-segmentation", "Bank Customer Segmentation"),
        ("barelydedicated/bank-customer-churn-modeling", "Bank Customer Churn"),
        ("mathchi/churn-for-bank-customers", "Bank Customer Churn Prediction"),
        
        # Customer transaction patterns
        ("volodymyrgavrysh/bank-marketing-campaigns-dataset", "Bank Marketing Campaigns"),
        ("henriqueyamahata/bank-marketing", "Bank Marketing Dataset"),
        ("janiobachmann/bank-marketing-dataset", "Bank Marketing Analysis"),
        
        # Customer segmentation and behavior
        ("vjchoudhary7/customer-segmentation-tutorial-in-python", "Customer Segmentation Tutorial"),
        ("abisheksudarshan/customer-segmentation", "Customer Segmentation Analysis"),
        
        # Banking transactions and accounts
        ("ealaxi/banksim1", "Bank Simulation Dataset"),
        ("ntnu-testimon/banksim1", "BankSim Banking Transactions"),
        
        # Credit and loans (good for CLV)
        ("laotse/credit-risk-dataset", "Credit Risk Dataset"),
        ("rikdifos/credit-card-approval-prediction", "Credit Card Approval"),
        
        # Retail banking
        ("sinderpreet/retail-banking-dataset", "Retail Banking Dataset"),
        ("blastchar/telco-customer-churn", "Telco Customer Churn"),  # Similar business model
    ]
    
    successful_dataset = None
    best_score = 0
    
    for dataset_id, description in clv_datasets:
        try:
            print(f"\n🔍 Analyzing: {description}")
            print(f"   Dataset ID: {dataset_id}")
            
            # Download dataset
            path = kagglehub.dataset_download(dataset_id)
            print(f"   ✅ Downloaded to: {path}")
            
            # Find data files
            files = []
            for root, dirs, filenames in os.walk(path):
                for filename in filenames:
                    if filename.lower().endswith(('.csv', '.xlsx', '.xls')):
                        files.append(os.path.join(root, filename))
            
            if not files:
                print(f"   ❌ No data files found")
                continue
            
            print(f"   📁 Found {len(files)} data file(s)")
            
            # Analyze all files and pick the best one
            best_file = None
            best_file_score = 0
            
            for file_path in files:
                try:
                    print(f"     📄 Analyzing: {os.path.basename(file_path)}")
                    
                    # Load the dataset
                    if file_path.lower().endswith('.csv'):
                        df_temp = pd.read_csv(file_path, nrows=1000)  # Sample first 1000 rows
                    else:
                        df_temp = pd.read_excel(file_path, nrows=1000)
                    
                    # Evaluate CLV suitability
                    score = evaluate_clv_suitability_detailed(df_temp, os.path.basename(file_path))
                    print(f"     ⭐ CLV Score: {score}/10")
                    
                    if score > best_file_score:
                        best_file_score = score
                        best_file = file_path
                        
                except Exception as e:
                    print(f"     ❌ Error analyzing file: {e}")
                    continue
            
            if best_file and best_file_score >= 6:
                # Load the full dataset
                print(f"   🎯 Loading best file: {os.path.basename(best_file)}")
                if best_file.lower().endswith('.csv'):
                    df = pd.read_csv(best_file)
                else:
                    df = pd.read_excel(best_file)
                
                print(f"   📊 Full dataset shape: {df.shape}")
                print(f"   📋 Columns: {list(df.columns)}")
                print(f"   ⭐ Final CLV Suitability Score: {best_file_score}/10")
                
                if best_file_score > best_score:
                    best_score = best_file_score
                    successful_dataset = {
                        'dataset_id': dataset_id,
                        'description': description,
                        'dataframe': df,
                        'file_path': best_file,
                        'suitability_score': best_file_score
                    }
                    print(f"   🏆 NEW BEST DATASET!")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    return successful_dataset

def evaluate_clv_suitability_detailed(df, filename):
    """
    Detailed evaluation of dataset suitability for CLV analysis
    """
    score = 0
    reasons = []
    
    # Customer identifier (essential for CLV)
    customer_cols = ['customer_id', 'customerid', 'cust_id', 'user_id', 'userid', 'id', 'customer']
    has_customer_id = any(col.lower() in [c.lower() for c in df.columns] for col in customer_cols)
    if has_customer_id:
        score += 3
        reasons.append("✅ Has customer identifier")
    else:
        reasons.append("❌ No customer identifier")
    
    # Transaction/interaction dates (essential for recency)
    date_cols = ['date', 'time', 'timestamp', 'transaction_date', 'created_at', 'last_transaction', 'signup_date']
    has_date = any(col.lower() in [c.lower() for c in df.columns] for col in date_cols)
    if has_date:
        score += 2
        reasons.append("✅ Has date/time information")
    else:
        reasons.append("❌ No date/time information")
    
    # Monetary values (essential for monetary component)
    money_cols = ['amount', 'value', 'price', 'cost', 'revenue', 'balance', 'income', 'salary', 'loan', 'deposit']
    has_money = any(col.lower() in [c.lower() for c in df.columns] for col in money_cols)
    if has_money:
        score += 3
        reasons.append("✅ Has monetary values")
    else:
        reasons.append("❌ No monetary values")
    
    # Banking-specific indicators
    banking_cols = ['account', 'bank', 'credit', 'loan', 'deposit', 'balance', 'transaction', 'payment']
    has_banking = any(col.lower() in [c.lower() for c in df.columns] for col in banking_cols)
    if has_banking:
        score += 1
        reasons.append("✅ Has banking-related columns")
    
    # Customer behavior indicators
    behavior_cols = ['frequency', 'duration', 'tenure', 'products', 'services', 'campaign', 'response']
    has_behavior = any(col.lower() in [c.lower() for c in df.columns] for col in behavior_cols)
    if has_behavior:
        score += 1
        reasons.append("✅ Has customer behavior indicators")
    
    # Dataset size
    if len(df) > 1000:
        score += 1
        reasons.append(f"✅ Good size: {len(df)} rows")
    
    print(f"       Evaluation reasons: {'; '.join(reasons)}")
    return score

def create_realistic_bank_clv_data():
    """
    Create a realistic banking dataset specifically designed for CLV analysis
    """
    print("\n🏦 Creating realistic banking CLV dataset...")
    
    np.random.seed(42)
    
    # Customer segments for CLV analysis
    segments = {
        'VIP': {'prob': 0.05, 'avg_products': 5, 'avg_balance': 100000, 'churn_rate': 0.02},
        'Premium': {'prob': 0.15, 'avg_products': 3, 'avg_balance': 50000, 'churn_rate': 0.05},
        'Gold': {'prob': 0.25, 'avg_products': 2, 'avg_balance': 25000, 'churn_rate': 0.10},
        'Standard': {'prob': 0.35, 'avg_products': 1.5, 'avg_balance': 10000, 'churn_rate': 0.15},
        'Basic': {'prob': 0.20, 'avg_products': 1, 'avg_balance': 3000, 'churn_rate': 0.25}
    }
    
    # Generate customer base
    n_customers = 2000
    customers_data = []
    
    for i in range(n_customers):
        # Assign segment
        segment = np.random.choice(list(segments.keys()), 
                                 p=[segments[s]['prob'] for s in segments.keys()])
        
        # Customer acquisition date (spread over 5 years)
        acquisition_date = datetime(2019, 1, 1) + timedelta(days=np.random.randint(0, 1825))
        
        # Customer characteristics
        age = max(18, int(np.random.normal(45, 15)))
        annual_income = max(20000, int(np.random.normal(60000, 30000)))
        
        customer = {
            'customer_id': f"BANK_{i+1:06d}",
            'segment': segment,
            'acquisition_date': acquisition_date,
            'age': age,
            'annual_income': annual_income,
            'city': np.random.choice(['Warsaw', 'Krakow', 'Gdansk', 'Wroclaw', 'Poznan']),
            'num_products': max(1, int(np.random.poisson(segments[segment]['avg_products']))),
            'account_balance': max(0, np.random.normal(segments[segment]['avg_balance'], 
                                                     segments[segment]['avg_balance'] * 0.4)),
            'credit_score': np.random.randint(300, 850),
            'is_active': np.random.random() > segments[segment]['churn_rate']
        }
        customers_data.append(customer)
    
    # Generate transaction history for CLV calculation
    transactions_data = []
    products = [
        ('Current_Account', 0, 10),      # Monthly fee
        ('Savings_Account', 50, 5),      # Interest earned, small fee
        ('Credit_Card', -200, 25),       # Average spend, fees
        ('Personal_Loan', -500, 50),     # Monthly payment, profit
        ('Mortgage', -1500, 200),        # Monthly payment, high profit
        ('Investment', 100, 30),         # Returns, management fee
        ('Insurance', -100, 40),         # Premium, profit
    ]
    
    current_date = datetime(2024, 12, 1)
    
    for customer in customers_data:
        customer_id = customer['customer_id']
        start_date = customer['acquisition_date']
        
        # Calculate months since acquisition
        months_active = max(1, (current_date - start_date).days // 30)
        
        if not customer['is_active']:
            # Inactive customers stopped transactions some time ago
            months_active = max(1, months_active - np.random.randint(1, 12))
        
        # Generate monthly transactions
        for month in range(months_active):
            transaction_date = start_date + timedelta(days=month * 30 + np.random.randint(0, 30))
            
            # Each customer can have multiple products
            for product_name, avg_flow, profit_margin in products[:customer['num_products']]:
                if np.random.random() < 0.9:  # 90% chance of monthly activity per product
                    
                    # Calculate monthly cash flow and bank profit
                    cash_flow = avg_flow + np.random.normal(0, abs(avg_flow) * 0.3)
                    bank_profit = profit_margin + np.random.normal(0, profit_margin * 0.2)
                    
                    # Adjust based on customer segment
                    segment_multiplier = {
                        'VIP': 3.0, 'Premium': 2.0, 'Gold': 1.5, 'Standard': 1.0, 'Basic': 0.7
                    }[customer['segment']]
                    
                    cash_flow *= segment_multiplier
                    bank_profit *= segment_multiplier
                    
                    transaction = {
                        'customer_id': customer_id,
                        'transaction_date': transaction_date,
                        'product_type': product_name,
                        'monthly_cash_flow': round(cash_flow, 2),
                        'bank_profit': round(bank_profit, 2),
                        'customer_segment': customer['segment'],
                        'transaction_month': transaction_date.strftime('%Y-%m')
                    }
                    transactions_data.append(transaction)
    
    # Create DataFrames
    customers_df = pd.DataFrame(customers_data)
    transactions_df = pd.DataFrame(transactions_data)
    
    # Calculate CLV metrics for each customer
    clv_data = []
    for customer_id in customers_df['customer_id'].unique():
        customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id]
        customer_info = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
        
        if len(customer_transactions) > 0:
            # Calculate RFM metrics
            last_transaction = pd.to_datetime(customer_transactions['transaction_date']).max()
            recency = (current_date - last_transaction).days
            frequency = len(customer_transactions)
            monetary = customer_transactions['bank_profit'].sum()
            
            # Calculate lifetime value (total profit generated)
            lifetime_value = customer_transactions['bank_profit'].sum()
            
            # Calculate average monthly value
            months_active = max(1, (current_date - customer_info['acquisition_date']).days // 30)
            avg_monthly_value = lifetime_value / months_active if months_active > 0 else 0
            
            clv_record = {
                'customer_id': customer_id,
                'segment': customer_info['segment'],
                'acquisition_date': customer_info['acquisition_date'],
                'recency_days': recency,
                'frequency': frequency,
                'monetary_value': round(monetary, 2),
                'lifetime_value': round(lifetime_value, 2),
                'avg_monthly_value': round(avg_monthly_value, 2),
                'months_active': months_active,
                'is_active': customer_info['is_active'],
                'num_products': customer_info['num_products'],
                'account_balance': customer_info['account_balance']
            }
            clv_data.append(clv_record)
    
    clv_df = pd.DataFrame(clv_data)
    
    print(f"   👥 Generated {len(customers_df)} customers")
    print(f"   💳 Generated {len(transactions_df)} transactions")
    print(f"   📊 Generated {len(clv_df)} CLV records")
    print(f"   📅 Date range: {transactions_df['transaction_date'].min()} to {transactions_df['transaction_date'].max()}")
    print(f"   💰 Total bank profit: ${clv_df['lifetime_value'].sum():,.2f}")
    
    return clv_df, transactions_df, customers_df

def main():
    """
    Main function to find or create suitable banking CLV dataset
    """
    print("🏦 BANKING CLV PROJECT - FINDING SUITABLE DATA")
    print("="*60)
    
    # Try to find real banking datasets suitable for CLV
    result = search_clv_suitable_datasets()
    
    if result and result['suitability_score'] >= 7:
        print(f"\n✅ Found excellent dataset: {result['description']}")
        print(f"   Score: {result['suitability_score']}/10")
        main_df = result['dataframe']
        source = f"Kaggle: {result['dataset_id']}"
        
        # Save the found dataset
        os.makedirs("data", exist_ok=True)
        main_df.to_csv("data/banking_clv_dataset.csv", index=False)
        
    else:
        print("\n🔄 No highly suitable CLV datasets found.")
        print("Creating synthetic banking CLV dataset optimized for analysis...")
        
        clv_df, transactions_df, customers_df = create_realistic_bank_clv_data()
        main_df = clv_df
        source = "Synthetic banking CLV data"
        
        # Save all datasets
        os.makedirs("data", exist_ok=True)
        clv_df.to_csv("data/banking_clv_dataset.csv", index=False)
        transactions_df.to_csv("data/banking_transactions.csv", index=False)
        customers_df.to_csv("data/banking_customers.csv", index=False)
        
        print(f"\n💾 Saved datasets:")
        print(f"   - data/banking_clv_dataset.csv (main CLV data)")
        print(f"   - data/banking_transactions.csv (transaction history)")
        print(f"   - data/banking_customers.csv (customer profiles)")
    
    # Save metadata
    with open("data/banking_clv_info.txt", "w") as f:
        f.write(f"Banking CLV Dataset Info\\n")
        f.write(f"Created: {datetime.now()}\\n")
        f.write(f"Source: {source}\\n")
        f.write(f"Main dataset shape: {main_df.shape}\\n")
        f.write(f"Columns: {list(main_df.columns)}\\n")
        if 'customer_id' in main_df.columns:
            f.write(f"Unique customers: {main_df['customer_id'].nunique()}\\n")
        if 'lifetime_value' in main_df.columns:
            f.write(f"Total CLV: ${main_df['lifetime_value'].sum():,.2f}\\n")
            f.write(f"Average CLV: ${main_df['lifetime_value'].mean():.2f}\\n")
    
    # Display overview
    print("\\n" + "="*50)
    print("BANKING CLV DATASET OVERVIEW")
    print("="*50)
    print(f"Shape: {main_df.shape}")
    print(f"Columns: {list(main_df.columns)}")
    print("\\nFirst 5 rows:")
    print(main_df.head())
    
    if 'lifetime_value' in main_df.columns:
        print(f"\\n💰 Customer Lifetime Value Statistics:")
        print(main_df['lifetime_value'].describe())
        print(f"\\n📊 CLV Distribution by segment:")
        if 'segment' in main_df.columns:
            segment_clv = main_df.groupby('segment')['lifetime_value'].agg(['count', 'mean', 'sum'])
            print(segment_clv)
    
    return main_df

if __name__ == "__main__":
    dataset = main()