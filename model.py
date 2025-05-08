import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

FEATURE_COLUMNS_TO_DROP = ['trial_id', 'label'] # Columns not used directly as input features
TARGET_COLUMN = 'label'

def split_data(features_df, test_size=0.2, random_state=42):
    """Splits the features DataFrame into training and testing sets."""
    X = features_df.drop(columns=FEATURE_COLUMNS_TO_DROP)
    y = features_df[TARGET_COLUMN]
    
    # Check for NaNs before scaling and training
    if X.isnull().sum().any():
        print("Warning: NaN values found in features X. Filling with column mean.")
        # Simple mean imputation. Other strategies might be better.
        for col in X.columns[X.isnull().any()]:
            X[col] = X[col].fillna(X[col].mean())
            
    # Feature scaling can help some models, although Random Forest is less sensitive.
    # It's a good practice to include it.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler # Return the scaler for later use if needed

def train_random_forest(X_train, y_train, random_state=42, n_estimators=100, class_weight=None):
    """Trains a RandomForestClassifier model."""
    print(f"Training RandomForestClassifier with n_estimators={n_estimators}, class_weight={class_weight}")
    # Try using class_weight='balanced' due to slight class imbalance.
    # Or 'balanced_subsample' if bootstrap=True (default)
    model = RandomForestClassifier(n_estimators=n_estimators, 
                                 random_state=random_state, 
                                 class_weight=class_weight if class_weight else 'balanced_subsample',
                                 n_jobs=-1) # Use all processors
    model.fit(X_train, y_train)
    print("Training completed.")
    return model

def predict(model, X_test):
    """Makes predictions using the trained model."""
    predictions = model.predict(X_test)
    predict_proba = model.predict_proba(X_test) # For ROC AUC or other probability-based metrics
    return predictions, predict_proba


if __name__ == '__main__':
    print("This script defines model functions. To test, run main.py.")
    # Example of how it could be used (requires feature_engineering.py and data_loader.py):
    try:
        from data_loader import load_all_data
        from feature_engineering import process_all_data_to_features
        
        print("Loading raw data...")
        raw_data = load_all_data()
        if not raw_data.empty:
            print("Processing for features...")
            feature_data = process_all_data_to_features(raw_data)
            if not feature_data.empty:
                print("Splitting data...")
                X_train, X_test, y_train, y_test, _ = split_data(feature_data)
                print(f"Train: {X_train.shape}, Test: {X_test.shape}")
                
                print("Training Random Forest model...")
                rf_model = train_random_forest(X_train, y_train, class_weight='balanced')
                
                print("Making predictions...")
                predictions, _ = predict(rf_model, X_test)
                print("Sample predictions:", predictions[:10])
                
                # A simple evaluation here (evaluation.py will do more)
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(y_test, predictions)
                print(f"Accuracy on test set: {acc:.4f}")
            else:
                print("Features dataset is empty.")
        else:
            print("Raw dataset is empty.")
            
    except ImportError as ie:
        print(f"Import error: {ie}. Make sure the scripts are in the correct directory.")
    except Exception as e:
        print(f"An error occurred during model.py testing: {e}")
        import traceback
        traceback.print_exc() 