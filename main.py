# main.py
import time

# Import functions from other modules
from data_loader import load_all_data
from feature_engineering import process_all_data_to_features
from model import split_data, train_random_forest, predict
from evaluation import evaluate_model

# Configurable Parameters (could come from command line arguments or config file)
BASE_DATA_DIR = '.' # Base directory where sub1, sub2, ... folders are located
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS_RF = 150 # Number of trees in the Random Forest
CLASS_WEIGHT_RF = 'balanced' # or None, 'balanced_subsample'

def main():
    start_time = time.time()
    print("--- Starting Fall Detection Pipeline ---")

    # 1. Load Raw Data
    print("\n[Step 1/5] Loading raw data...")
    raw_data = load_all_data(base_dir=BASE_DATA_DIR)
    if raw_data.empty:
        print("Error: Failed to load raw data. Aborting.")
        return
    print(f"Raw data loaded. Shape: {raw_data.shape}")
    load_time = time.time()
    print(f"Loading time: {load_time - start_time:.2f} seconds")

    # 2. Feature Engineering
    print("\n[Step 2/5] Starting feature engineering...")
    feature_data = process_all_data_to_features(raw_data)
    if feature_data.empty:
        print("Error: Failed to generate features. Aborting.")
        return
    print(f"Features generated. Shape: {feature_data.shape}")

    # --- Add Saving Block ---
    print("\nSaving features dataset to CSV file...")
    save_filename = "processed_features_right_ankle.csv"
    try:
        feature_data.to_csv(save_filename, index=False)
        print(f"Features dataset successfully saved to: {save_filename}")
    except Exception as e:
        print(f"Error saving features dataset: {e}")
    # --- End of Saving Block ---

    feature_time = time.time()
    print(f"Feature engineering time: {feature_time - load_time:.2f} seconds")
    
    # Optional: Save/Load features to avoid recalculation (now implemented above)
    # feature_data.to_csv("processed_features_right_ankle.csv", index=False) 
    # feature_data = pd.read_csv("processed_features_right_ankle.csv")

    # 3. Split Data
    print("\n[Step 3/5] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test, scaler = split_data(
        feature_data, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_STATE
    )
    print(f"Data split. Train: {X_train.shape}, Test: {X_test.shape}")
    split_time = time.time()
    print(f"Split time: {split_time - feature_time:.2f} seconds")

    # 4. Train Model
    print("\n[Step 4/5] Training the Random Forest model...")
    rf_model = train_random_forest(
        X_train, 
        y_train, 
        random_state=RANDOM_STATE, 
        n_estimators=N_ESTIMATORS_RF,
        class_weight=CLASS_WEIGHT_RF
    )
    train_time = time.time()
    print(f"Training time: {train_time - split_time:.2f} seconds")
    
    # Optional: Save trained model
    # import joblib
    # joblib.dump(rf_model, 'random_forest_fall_detection.joblib')
    # rf_model = joblib.load('random_forest_fall_detection.joblib')

    # 5. Evaluate Model
    print("\n[Step 5/5] Evaluating the model on the test set...")
    predictions, predict_proba = predict(rf_model, X_test)
    evaluate_model(y_test, predictions, predict_proba[:, 1], model_name="Random Forest Baseline") # Pass probability of class 1
    eval_time = time.time()
    print(f"Evaluation time: {eval_time - train_time:.2f} seconds")

    end_time = time.time()
    print(f"\n--- Pipeline Completed in {end_time - start_time:.2f} seconds ---")

if __name__ == '__main__':
    # Check if matplotlib and seaborn are installed (needed for evaluation.py)
    try:
        import matplotlib
        import seaborn
        import sklearn 
    except ImportError as e:
        print(f"Error: Required library ({e.name}) not found.")
        print("Please install dependencies: pip install matplotlib seaborn scikit-learn")
    else:
        main() 