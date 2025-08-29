import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os


warnings.filterwarnings('ignore')

plt.style.use('default')  # Use default style since seaborn-v0_8 might not be available
sns.set_palette("husl")

class TelecomChurnPredictor:
    """
    A comprehensive class for telecom customer churn prediction using Gradient Boosting.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the TelecomChurnPredictor.
        
        Args:
            data_path (str): Path to the dataset file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_model = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, data_path=None):
        """
        Load and preprocess the Telco Customer Churn dataset.
        
        Args:
            data_path (str): Path to the dataset file
        """
        print("=" * 60)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("=" * 60)
        
        # Use provided path or class attribute
        if data_path:
            self.data_path = data_path
        
        try:
           
            if self.data_path and os.path.exists(self.data_path):
                print(f"Loading data from: {self.data_path}")
                self.df = pd.read_csv(self.data_path)
            else:
               
                print("Dataset file not found. Creating sample dataset structure...")
                print("Note: In production, ensure the Telco Customer Churn dataset is available.")
                
                
                np.random.seed(42)
                n_samples = 1000
                
                sample_data = {
                    'customerID': [f'customer_{i}' for i in range(n_samples)],
                    'gender': np.random.choice(['Male', 'Female'], n_samples),
                    'SeniorCitizen': np.random.choice([0, 1], n_samples),
                    'Partner': np.random.choice(['Yes', 'No'], n_samples),
                    'Dependents': np.random.choice(['Yes', 'No'], n_samples),
                    'tenure': np.random.randint(1, 73, n_samples),
                    'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
                    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
                    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
                    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
                    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
                    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
                    'MonthlyCharges': np.random.uniform(18.0, 120.0, n_samples),
                    'TotalCharges': [''] * n_samples,  # Will be calculated
                    'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
                }
                
                
                for i in range(n_samples):
                    if np.random.random() < 0.05:  # 5% missing values
                        sample_data['TotalCharges'][i] = ''
                    else:
                        sample_data['TotalCharges'][i] = str(round(sample_data['MonthlyCharges'][i] * sample_data['tenure'][i], 2))
                
                self.df = pd.DataFrame(sample_data)
                print(f"Created sample dataset with {n_samples} records")
            
            print(f"Dataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            print("\nDataset Info:")
            print(self.df.info())
            
            print("\nHandling missing values...")
            missing_before = self.df['TotalCharges'].isnull().sum()
            empty_strings = (self.df['TotalCharges'] == '').sum()
            print(f"Missing values in TotalCharges: {missing_before}")
            print(f"Empty strings in TotalCharges: {empty_strings}")
            
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            
            median_total_charges = self.df['TotalCharges'].median()
            self.df['TotalCharges'].fillna(median_total_charges, inplace=True)
            
            missing_after = self.df['TotalCharges'].isnull().sum()
            print(f"Missing values after cleaning: {missing_after}")
            
            self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
            
            print("\nData preprocessing completed successfully!")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def exploratory_data_analysis(self):
        """
        Conduct comprehensive exploratory data analysis.
        """
        print("\n" + "=" * 60)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        print("Dataset Overview:")
        print(f"Total customers: {len(self.df)}")
        print(f"Churned customers: {self.df['Churn'].sum()}")
        print(f"Churn rate: {self.df['Churn'].mean():.2%}")
        
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Telecom Customer Churn - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        churn_counts = self.df['Churn'].value_counts()
        axes[0, 0].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
                       colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Churn Distribution')
        
        contract_churn = pd.crosstab(self.df['Contract'], self.df['Churn'], normalize='index')
        contract_churn.plot(kind='bar', ax=axes[0, 1], color=['lightblue', 'lightcoral'])
        axes[0, 1].set_title('Churn Rate by Contract Type')
        axes[0, 1].set_xlabel('Contract Type')
        axes[0, 1].set_ylabel('Proportion')
        axes[0, 1].legend(['No Churn', 'Churn'])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        sns.violinplot(data=self.df, x='Churn', y='MonthlyCharges', ax=axes[0, 2])
        axes[0, 2].set_title('Monthly Charges Distribution by Churn')
        axes[0, 2].set_xticklabels(['No Churn', 'Churn'])
        
        sns.violinplot(data=self.df, x='Churn', y='tenure', ax=axes[1, 0])
        axes[1, 0].set_title('Tenure Distribution by Churn')
        axes[1, 0].set_xticklabels(['No Churn', 'Churn'])
        
        internet_churn = pd.crosstab(self.df['InternetService'], self.df['Churn'], normalize='index')
        internet_churn.plot(kind='bar', ax=axes[1, 1], color=['lightblue', 'lightcoral'])
        axes[1, 1].set_title('Churn Rate by Internet Service')
        axes[1, 1].set_xlabel('Internet Service')
        axes[1, 1].set_ylabel('Proportion')
        axes[1, 1].legend(['No Churn', 'Churn'])
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        payment_churn = pd.crosstab(self.df['PaymentMethod'], self.df['Churn'], normalize='index')
        payment_churn.plot(kind='bar', ax=axes[1, 2], color=['lightblue', 'lightcoral'])
        axes[1, 2].set_title('Churn Rate by Payment Method')
        axes[1, 2].set_xlabel('Payment Method')
        axes[1, 2].set_ylabel('Proportion')
        axes[1, 2].legend(['No Churn', 'Churn'])
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print("EDA visualizations saved to 'eda_analysis.png'")
        
        print("\nKey Insights from EDA:")
        
        contract_analysis = self.df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
        print(f"\nChurn rate by Contract:")
        for contract, rate in contract_analysis.items():
            print(f"  {contract}: {rate:.2%}")
        
        internet_analysis = self.df.groupby('InternetService')['Churn'].mean().sort_values(ascending=False)
        print(f"\nChurn rate by Internet Service:")
        for service, rate in internet_analysis.items():
            print(f"  {service}: {rate:.2%}")
        
        charges_analysis = self.df.groupby('Churn')['MonthlyCharges'].mean()
        print(f"\nAverage Monthly Charges:")
        print(f"  No Churn: ${charges_analysis[0]:.2f}")
        print(f"  Churn: ${charges_analysis[1]:.2f}")
        
    def prepare_features(self):
        """
        Prepare features for model training by applying one-hot encoding.
        """
        print("\n" + "=" * 60)
        print("STEP 3: FEATURE PREPARATION")
        print("=" * 60)
        
        df_features = self.df.copy()
        
        if 'customerID' in df_features.columns:
            df_features = df_features.drop('customerID', axis=1)
        
        y = df_features['Churn']
        X = df_features.drop('Churn', axis=1)
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        
        print("Applying one-hot encoding to categorical variables...")
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        print(f"Features before encoding: {X.shape[1]}")
        print(f"Features after encoding: {X_encoded.shape[1]}")
        
        self.feature_names = X_encoded.columns.tolist()
        
        print("Splitting data into training and testing sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Testing set size: {self.X_test.shape}")
        print(f"Training set churn rate: {self.y_train.mean():.2%}")
        print(f"Testing set churn rate: {self.y_test.mean():.2%}")
        
    def train_initial_model(self):
        """
        Train initial Gradient Boosting Classifier model.
        """
        print("\n" + "=" * 60)
        print("STEP 4: INITIAL MODEL TRAINING")
        print("=" * 60)
        
        print("Training Gradient Boosting Classifier...")
        self.model = GradientBoostingClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Initial model accuracy: {accuracy:.4f}")
        
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    def optimize_hyperparameters(self):
        """
        Optimize model hyperparameters using GridSearchCV.
        """
        print("\n" + "=" * 60)
        print("STEP 5: HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 4, 5],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10]
        }
        
        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        print(f"\nTotal combinations to test: {np.prod([len(v) for v in param_grid.values()])}")
        
        print("Performing GridSearchCV (this may take a while)...")
        grid_search = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid,
            cv=3,  # Reduced CV folds for faster execution
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
    def evaluate_model(self):
        """
        Evaluate the optimized model and generate visualizations.
        """
        print("\n" + "=" * 60)
        print("STEP 6: MODEL EVALUATION")
        print("=" * 60)
        
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Optimized model accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        sns.barplot(data=feature_importance, x='importance', y='feature', ax=axes[1])
        axes[1].set_title('Top 15 Feature Importances')
        axes[1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print("Model evaluation visualizations saved to 'model_evaluation.png'")
        
        print("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance.head(10).values, 1):
            print(f"{i:2d}. {feature}: {importance:.4f}")
        
        self.generate_insights(feature_importance)
        
    def generate_insights(self, feature_importance):
        """
        Generate actionable business insights from the model.
        
        Args:
            feature_importance (pd.DataFrame): Feature importance scores
        """
        print("\n" + "=" * 60)
        print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
        print("=" * 60)
        
        top_features = feature_importance.head(5)['feature'].tolist()
        
        print("Key Findings:")
        
        insights = {
            'MonthlyCharges': "Higher monthly charges are associated with increased churn risk. Consider reviewing pricing strategy.",
            'TotalCharges': "Customers with higher total charges show different churn patterns. Focus on long-term value customers.",
            'tenure': "Customer tenure is a strong predictor. Implement retention programs for new customers.",
            'Contract_Month-to-month': "Month-to-month contracts have higher churn rates. Incentivize longer-term contracts.",
            'Contract_Two year': "Two-year contracts reduce churn significantly. Promote longer commitments.",
            'InternetService_Fiber optic': "Fiber optic customers may have different churn patterns. Analyze service quality.",
            'PaymentMethod_Electronic check': "Electronic check payment method may indicate higher churn risk.",
            'PaperlessBilling_Yes': "Paperless billing preference may correlate with churn behavior.",
            'OnlineSecurity_No': "Lack of online security service may increase churn risk.",
            'TechSupport_No': "Customers without tech support are more likely to churn."
        }
        
        recommendations = []
        
        for i, feature in enumerate(top_features, 1):
            if feature in insights:
                print(f"{i}. {insights[feature]}")
                
                if 'Contract' in feature:
                    recommendations.append("Implement contract upgrade incentives")
                elif 'MonthlyCharges' in feature:
                    recommendations.append("Review pricing tiers and offer personalized discounts")
                elif 'tenure' in feature:
                    recommendations.append("Develop early customer engagement programs")
                elif 'TechSupport' in feature or 'OnlineSecurity' in feature:
                    recommendations.append("Promote additional service bundles")
        
        print("\nRecommended Actions:")
        for i, rec in enumerate(set(recommendations), 1):
            print(f"{i}. {rec}")
        
        print(f"\nModel Performance Summary:")
        print(f"- The model can predict customer churn with {accuracy_score(self.y_test, self.best_model.predict(self.X_test)):.1%} accuracy")
        print(f"- Focus on the top {len(top_features)} features for maximum impact")
        print(f"- Implement targeted retention strategies based on risk factors")
        
    def run_complete_pipeline(self, data_path=None):
        """
        Run the complete machine learning pipeline.
        
        Args:
            data_path (str): Path to the dataset file
        """
        try:
            print("TELECOM CUSTOMER CHURN PREDICTION PIPELINE")
            print("=" * 60)
            print("Using Gradient Boosting Classifier with comprehensive ML pipeline")
            print("=" * 60)
            
            self.load_and_preprocess_data(data_path)
            self.exploratory_data_analysis()
            self.prepare_features()
            self.train_initial_model()
            self.optimize_hyperparameters()
            self.evaluate_model()
            
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
        except Exception as e:
            print(f"Error in pipeline execution: {str(e)}")
            raise

def main():
    """
    Main function to execute the churn prediction pipeline.
    """
    predictor = TelecomChurnPredictor()
    
    data_path = "telco_customer_churn.csv"  # Update with your dataset path
    
    predictor.run_complete_pipeline(data_path)

if __name__ == "__main__":
    main()
