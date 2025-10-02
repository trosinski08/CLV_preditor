"""
Banking Data Cleaning Module for CLV Analysis
Comprehensive data cleaning and preprocessing for banking customer dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BankingDataCleaner:
    """
    Professional data cleaning class for banking CLV analysis
    """
    
    def __init__(self, data_path="data/banking_clv_dataset.csv"):
        """Initialize the data cleaner with dataset path"""
        self.data_path = data_path
        self.df_original = None
        self.df_cleaned = None
        self.cleaning_report = {
            'original_shape': None,
            'cleaned_shape': None,
            'missing_values': {},
            'outliers_removed': {},
            'data_types_fixed': [],
            'feature_engineering': [],
            'quality_score': None
        }
        
    def load_data(self):
        """Load the original banking dataset"""
        print("🔄 Loading banking dataset...")
        try:
            self.df_original = pd.read_csv(self.data_path)
            self.cleaning_report['original_shape'] = self.df_original.shape
            print(f"✅ Dataset loaded: {self.df_original.shape[0]:,} customers, {self.df_original.shape[1]} features")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def assess_data_quality(self):
        """Comprehensive data quality assessment"""
        print("\n🔍 ASSESSING DATA QUALITY")
        print("="*50)
        
        if self.df_original is None:
            print("❌ No data loaded")
            return
        
        df = self.df_original.copy()
        
        # Basic info
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Missing values analysis
        print(f"\n🔍 Missing Values Analysis:")
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        
        if total_missing > 0:
            print(f"⚠️  Total missing values: {total_missing:,}")
            for col, count in missing_summary.items():
                if count > 0:
                    percentage = (count / len(df)) * 100
                    print(f"   {col}: {count:,} ({percentage:.1f}%)")
                    self.cleaning_report['missing_values'][col] = {
                        'count': count, 'percentage': percentage
                    }
        else:
            print("✅ No missing values found")
        
        # Data types analysis
        print(f"\n📋 Data Types:")
        dtype_summary = df.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            print(f"   {dtype}: {count} columns")
        
        # Duplicates check
        duplicates = df.duplicated().sum()
        print(f"\n🔄 Duplicate Rows: {duplicates:,}")
        
        # Unique values analysis
        print(f"\n🔢 Unique Values per Column:")
        for col in df.columns:
            unique_count = df[col].nunique()
            unique_percentage = (unique_count / len(df)) * 100
            print(f"   {col}: {unique_count:,} ({unique_percentage:.1f}%)")
        
        # Basic statistics for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n📈 Numerical Columns Statistics:")
            stats = df[numerical_cols].describe()
            print(stats.round(2))
    
    def clean_customer_identifiers(self):
        """Clean and validate customer identifiers"""
        print("\n🔧 CLEANING CUSTOMER IDENTIFIERS")
        print("="*40)
        
        df = self.df_original.copy()
        
        # Check CustomerId
        if 'CustomerId' in df.columns:
            print(f"📋 Customer ID Analysis:")
            print(f"   Total customers: {df['CustomerId'].nunique():,}")
            print(f"   ID range: {df['CustomerId'].min()} - {df['CustomerId'].max()}")
            
            # Check for invalid customer IDs
            invalid_ids = df['CustomerId'].isnull().sum()
            if invalid_ids > 0:
                print(f"⚠️  Invalid customer IDs: {invalid_ids}")
                df = df.dropna(subset=['CustomerId'])
                self.cleaning_report['missing_values']['CustomerId_removed'] = invalid_ids
            else:
                print("✅ All customer IDs are valid")
            
            # Ensure customer ID is integer
            df['CustomerId'] = df['CustomerId'].astype(int)
            self.cleaning_report['data_types_fixed'].append('CustomerId to int')
        
        # Clean RowNumber (if exists and not needed)
        if 'RowNumber' in df.columns:
            print("🗑️  Removing RowNumber column (not needed for analysis)")
            df = df.drop('RowNumber', axis=1)
            self.cleaning_report['feature_engineering'].append('Removed RowNumber')
        
        return df
    
    def clean_demographic_data(self, df):
        """Clean demographic information"""
        print("\n👥 CLEANING DEMOGRAPHIC DATA")
        print("="*35)
        
        # Age validation
        if 'Age' in df.columns:
            print(f"📊 Age Analysis:")
            print(f"   Age range: {df['Age'].min()} - {df['Age'].max()} years")
            print(f"   Average age: {df['Age'].mean():.1f} years")
            
            # Check for invalid ages
            invalid_ages = ((df['Age'] < 18) | (df['Age'] > 100)).sum()
            if invalid_ages > 0:
                print(f"⚠️  Invalid ages (< 18 or > 100): {invalid_ages}")
                df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]
                self.cleaning_report['outliers_removed']['Age'] = invalid_ages
            else:
                print("✅ All ages are valid")
        
        # Geography standardization
        if 'Geography' in df.columns:
            print(f"\n🌍 Geography Analysis:")
            geo_counts = df['Geography'].value_counts()
            print("   Country distribution:")
            for country, count in geo_counts.items():
                print(f"      {country}: {count:,} customers")
            
            # Standardize country names
            df['Geography'] = df['Geography'].str.strip().str.title()
            self.cleaning_report['data_types_fixed'].append('Geography standardized')
        
        # Gender standardization
        if 'Gender' in df.columns:
            print(f"\n⚥ Gender Analysis:")
            gender_counts = df['Gender'].value_counts()
            for gender, count in gender_counts.items():
                print(f"   {gender}: {count:,} customers")
            
            # Standardize gender values
            df['Gender'] = df['Gender'].str.strip().str.title()
            self.cleaning_report['data_types_fixed'].append('Gender standardized')
        
        return df
    
    def clean_financial_data(self, df):
        """Clean financial information"""
        print("\n💰 CLEANING FINANCIAL DATA")
        print("="*30)
        
        # Account Balance validation
        if 'Balance' in df.columns:
            print(f"💳 Account Balance Analysis:")
            print(f"   Balance range: ${df['Balance'].min():,.2f} - ${df['Balance'].max():,.2f}")
            print(f"   Average balance: ${df['Balance'].mean():,.2f}")
            print(f"   Zero balances: {(df['Balance'] == 0).sum():,} customers")
            
            # Check for negative balances (might be valid for overdrafts)
            negative_balances = (df['Balance'] < 0).sum()
            if negative_balances > 0:
                print(f"⚠️  Negative balances: {negative_balances:,} (keeping as valid overdrafts)")
            
            # Check for extremely high balances (potential data errors)
            high_threshold = df['Balance'].quantile(0.999)
            extreme_balances = (df['Balance'] > high_threshold * 3).sum()
            if extreme_balances > 0:
                print(f"⚠️  Extremely high balances (>3x 99.9th percentile): {extreme_balances}")
                # Could choose to cap or investigate these
        
        # Estimated Salary validation
        if 'EstimatedSalary' in df.columns:
            print(f"\n💼 Estimated Salary Analysis:")
            print(f"   Salary range: ${df['EstimatedSalary'].min():,.2f} - ${df['EstimatedSalary'].max():,.2f}")
            print(f"   Average salary: ${df['EstimatedSalary'].mean():,.2f}")
            
            # Check for unrealistic salaries
            low_salary_threshold = 10000  # Below minimum wage
            high_salary_threshold = 500000  # Unrealistically high
            
            low_salaries = (df['EstimatedSalary'] < low_salary_threshold).sum()
            high_salaries = (df['EstimatedSalary'] > high_salary_threshold).sum()
            
            if low_salaries > 0:
                print(f"⚠️  Very low salaries (<${low_salary_threshold:,}): {low_salaries}")
            if high_salaries > 0:
                print(f"⚠️  Very high salaries (>${high_salary_threshold:,}): {high_salaries}")
        
        # Credit Score validation
        if 'CreditScore' in df.columns:
            print(f"\n📊 Credit Score Analysis:")
            print(f"   Credit score range: {df['CreditScore'].min()} - {df['CreditScore'].max()}")
            print(f"   Average credit score: {df['CreditScore'].mean():.0f}")
            
            # Check for invalid credit scores
            invalid_scores = ((df['CreditScore'] < 300) | (df['CreditScore'] > 850)).sum()
            if invalid_scores > 0:
                print(f"⚠️  Invalid credit scores (not 300-850): {invalid_scores}")
                df = df[(df['CreditScore'] >= 300) & (df['CreditScore'] <= 850)]
                self.cleaning_report['outliers_removed']['CreditScore'] = invalid_scores
            else:
                print("✅ All credit scores are valid")
        
        return df
    
    def clean_banking_behavior(self, df):
        """Clean banking behavior data"""
        print("\n🏦 CLEANING BANKING BEHAVIOR DATA")
        print("="*40)
        
        # Tenure validation
        if 'Tenure' in df.columns:
            print(f"⏰ Tenure Analysis:")
            print(f"   Tenure range: {df['Tenure'].min()} - {df['Tenure'].max()} years")
            print(f"   Average tenure: {df['Tenure'].mean():.1f} years")
            
            # Check for invalid tenure
            invalid_tenure = ((df['Tenure'] < 0) | (df['Tenure'] > 50)).sum()
            if invalid_tenure > 0:
                print(f"⚠️  Invalid tenure (< 0 or > 50 years): {invalid_tenure}")
                df = df[(df['Tenure'] >= 0) & (df['Tenure'] <= 50)]
                self.cleaning_report['outliers_removed']['Tenure'] = invalid_tenure
            else:
                print("✅ All tenure values are valid")
        
        # Number of Products validation
        if 'NumOfProducts' in df.columns:
            print(f"\n🛍️  Number of Products Analysis:")
            product_counts = df['NumOfProducts'].value_counts().sort_index()
            for products, count in product_counts.items():
                print(f"   {products} product(s): {count:,} customers")
            
            # Check for invalid product counts
            invalid_products = ((df['NumOfProducts'] < 1) | (df['NumOfProducts'] > 10)).sum()
            if invalid_products > 0:
                print(f"⚠️  Invalid product counts (< 1 or > 10): {invalid_products}")
                df = df[(df['NumOfProducts'] >= 1) & (df['NumOfProducts'] <= 10)]
                self.cleaning_report['outliers_removed']['NumOfProducts'] = invalid_products
            else:
                print("✅ All product counts are valid")
        
        # Binary flags validation
        binary_columns = ['HasCrCard', 'IsActiveMember', 'Exited']
        for col in binary_columns:
            if col in df.columns:
                unique_values = df[col].unique()
                if not all(val in [0, 1] for val in unique_values):
                    print(f"⚠️  {col} has non-binary values: {unique_values}")
                    # Convert to binary if needed
                    df[col] = df[col].astype(int)
                else:
                    print(f"✅ {col} is properly binary")
        
        return df
    
    def engineer_clv_features(self, df):
        """Engineer features specifically for CLV analysis"""
        print("\n🔬 ENGINEERING CLV FEATURES")
        print("="*35)
        
        # Create age groups
        if 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], 
                                   bins=[0, 30, 40, 50, 60, 100], 
                                   labels=['<30', '30-40', '40-50', '50-60', '60+'])
            print("✅ Created AgeGroup feature")
            self.cleaning_report['feature_engineering'].append('AgeGroup')
        
        # Create balance segments
        if 'Balance' in df.columns:
            df['BalanceSegment'] = pd.cut(df['Balance'], 
                                         bins=[0, 1, 50000, 100000, 200000, float('inf')],
                                         labels=['Zero', 'Low', 'Medium', 'High', 'Very High'])
            print("✅ Created BalanceSegment feature")
            self.cleaning_report['feature_engineering'].append('BalanceSegment')
        
        # Create tenure segments
        if 'Tenure' in df.columns:
            df['TenureSegment'] = pd.cut(df['Tenure'], 
                                        bins=[0, 2, 5, 8, float('inf')],
                                        labels=['New', 'Growing', 'Mature', 'Loyal'])
            print("✅ Created TenureSegment feature")
            self.cleaning_report['feature_engineering'].append('TenureSegment')
        
        # Create credit score segments
        if 'CreditScore' in df.columns:
            df['CreditSegment'] = pd.cut(df['CreditScore'], 
                                        bins=[0, 580, 670, 740, 800, 850],
                                        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
            print("✅ Created CreditSegment feature")
            self.cleaning_report['feature_engineering'].append('CreditSegment')
        
        # Create customer value indicator
        if all(col in df.columns for col in ['Balance', 'NumOfProducts', 'Tenure']):
            # Simple customer value score
            df['CustomerValue'] = (
                (df['Balance'] / df['Balance'].max() * 0.5) +
                (df['NumOfProducts'] / df['NumOfProducts'].max() * 0.3) +
                (df['Tenure'] / df['Tenure'].max() * 0.2)
            )
            print("✅ Created CustomerValue feature")
            self.cleaning_report['feature_engineering'].append('CustomerValue')
        
        # Create CLV components for analysis
        if 'Balance' in df.columns:
            df['MonetaryValue'] = df['Balance']
            print("✅ Set MonetaryValue = Balance")
        
        if 'NumOfProducts' in df.columns:
            df['Frequency'] = df['NumOfProducts']
            print("✅ Set Frequency = NumOfProducts")
        
        if 'Tenure' in df.columns:
            # Recency: lower values = more recent (invert tenure)
            df['Recency'] = df['Tenure'].max() - df['Tenure'] + 1
            print("✅ Created Recency = inverted Tenure")
        
        return df
    
    def calculate_data_quality_score(self, df):
        """Calculate overall data quality score"""
        print("\n📊 CALCULATING DATA QUALITY SCORE")
        print("="*40)
        
        score = 0
        max_score = 100
        
        # Completeness (40 points)
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        completeness_score = max(0, 40 - (missing_percentage * 2))
        score += completeness_score
        print(f"📋 Completeness: {completeness_score:.1f}/40 (Missing: {missing_percentage:.1f}%)")
        
        # Consistency (30 points)
        consistency_issues = 0
        # Check for data type consistency
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dtype == 'object':
                consistency_issues += 1
        
        consistency_score = max(0, 30 - (consistency_issues * 5))
        score += consistency_score
        print(f"🔄 Consistency: {consistency_score:.1f}/30")
        
        # Validity (20 points)
        validity_issues = len(self.cleaning_report['outliers_removed'])
        validity_score = max(0, 20 - (validity_issues * 3))
        score += validity_score
        print(f"✅ Validity: {validity_score:.1f}/20")
        
        # Uniqueness (10 points)
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        uniqueness_score = max(0, 10 - (duplicate_percentage * 2))
        score += uniqueness_score
        print(f"🔢 Uniqueness: {uniqueness_score:.1f}/10 (Duplicates: {duplicate_percentage:.1f}%)")
        
        self.cleaning_report['quality_score'] = score
        print(f"\\n🏆 OVERALL DATA QUALITY SCORE: {score:.1f}/100")
        
        if score >= 90:
            print("🌟 Excellent data quality!")
        elif score >= 80:
            print("👍 Good data quality")
        elif score >= 70:
            print("⚠️  Acceptable data quality")
        else:
            print("❌ Poor data quality - needs attention")
        
        return score
    
    def clean_data(self, save_cleaned=True):
        """Main data cleaning pipeline"""
        print("🧹 STARTING COMPREHENSIVE DATA CLEANING")
        print("="*60)
        
        if not self.load_data():
            return False
        
        # Assess initial data quality
        self.assess_data_quality()
        
        # Clean data step by step
        df = self.clean_customer_identifiers()
        df = self.clean_demographic_data(df)
        df = self.clean_financial_data(df)
        df = self.clean_banking_behavior(df)
        df = self.engineer_clv_features(df)
        
        # Final quality assessment
        quality_score = self.calculate_data_quality_score(df)
        
        # Save cleaned data
        self.df_cleaned = df
        self.cleaning_report['cleaned_shape'] = df.shape
        
        if save_cleaned:
            output_path = "data/banking_clv_cleaned.csv"
            df.to_csv(output_path, index=False)
            print(f"\\n💾 Cleaned dataset saved: {output_path}")
        
        # Print final summary
        self.print_cleaning_summary()
        
        return True
    
    def print_cleaning_summary(self):
        """Print comprehensive cleaning summary"""
        print("\n" + "="*60)
        print("📋 DATA CLEANING SUMMARY")
        print("="*60)
        
        report = self.cleaning_report
        
        print(f"📊 Dataset Transformation:")
        print(f"   Original: {report['original_shape'][0]:,} rows × {report['original_shape'][1]} columns")
        print(f"   Cleaned:  {report['cleaned_shape'][0]:,} rows × {report['cleaned_shape'][1]} columns")
        
        if report['missing_values']:
            print(f"\\n🔍 Missing Values Handled:")
            for col, info in report['missing_values'].items():
                if isinstance(info, dict):
                    print(f"   {col}: {info['count']:,} ({info['percentage']:.1f}%)")
        
        if report['outliers_removed']:
            print(f"\\n🎯 Outliers Removed:")
            for col, count in report['outliers_removed'].items():
                print(f"   {col}: {count:,} records")
        
        if report['data_types_fixed']:
            print(f"\\n🔧 Data Types Fixed:")
            for fix in report['data_types_fixed']:
                print(f"   ✅ {fix}")
        
        if report['feature_engineering']:
            print(f"\\n🔬 Features Engineered:")
            for feature in report['feature_engineering']:
                print(f"   ✅ {feature}")
        
        print(f"\\n🏆 Final Data Quality Score: {report['quality_score']:.1f}/100")
        
        print(f"\\n✅ Data cleaning completed successfully!")
        print("   Ready for RFM analysis and CLV modeling")

def main():
    """Main function to run data cleaning"""
    cleaner = BankingDataCleaner()
    success = cleaner.clean_data()
    
    if success:
        print("\\n🎉 Data cleaning pipeline completed successfully!")
        return cleaner.df_cleaned
    else:
        print("\\n❌ Data cleaning failed!")
        return None

if __name__ == "__main__":
    cleaned_data = main()