import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

SAMPLING_FREQUENCY = 128  # Hz
WINDOW_SECONDS = 2  # seconds
STEP_SECONDS = 1    # seconds

WINDOW_SAMPLES = int(SAMPLING_FREQUENCY * WINDOW_SECONDS) # 256 samples
STEP_SAMPLES = int(SAMPLING_FREQUENCY * STEP_SECONDS)   # 128 samples

# Base sensor columns. Magnitudes will be added dynamically.
BASE_SENSOR_COLUMNS = [
    'r_acc_x', 'r_acc_y', 'r_acc_z',
    'r_gyro_x', 'r_gyro_y', 'r_gyro_z',
    'r_mag_x', 'r_mag_y', 'r_mag_z'
]

# This list will be populated by calculate_magnitudes and used to build
# the final list of columns for windowing.
# No longer a 'global' implicitly modified and read by other functions.
# _created_magnitude_columns = [] # Commented because it's no longer globally managed this way


def calculate_magnitudes(df):
    """Calculates the magnitudes of acceleration, gyroscope, and magnetometer vectors.
    Returns the DataFrame with the magnitude columns added and a list of created magnitude columns.
    """
    df_out = df.copy()
    created_magnitude_cols = []
    
    acc_cols = ['r_acc_x', 'r_acc_y', 'r_acc_z']
    gyro_cols = ['r_gyro_x', 'r_gyro_y', 'r_gyro_z']
    mag_cols = ['r_mag_x', 'r_mag_y', 'r_mag_z']

    if all(col in df_out.columns for col in acc_cols):
        df_out['r_acc_mag'] = np.sqrt((df_out[acc_cols]**2).sum(axis=1))
        created_magnitude_cols.append('r_acc_mag')
        
    if all(col in df_out.columns for col in gyro_cols):
        df_out['r_gyro_mag'] = np.sqrt((df_out[gyro_cols]**2).sum(axis=1))
        created_magnitude_cols.append('r_gyro_mag')

    if all(col in df_out.columns for col in mag_cols):
        df_out['r_mag_mag'] = np.sqrt((df_out[mag_cols]**2).sum(axis=1))
        created_magnitude_cols.append('r_mag_mag')
        
    return df_out, created_magnitude_cols


def create_features_for_window(window_data, cols_for_features): # cols_for_features is passed as argument
    """Calculates statistical features for a data window."""
    features = {}
    for col in cols_for_features:
        if col in window_data.columns:
            series = window_data[col]
            features[f'{col}_mean'] = series.mean()
            features[f'{col}_std'] = series.std()
            features[f'{col}_min'] = series.min()
            features[f'{col}_max'] = series.max()
            features[f'{col}_range'] = series.max() - series.min()
            features[f'{col}_median'] = series.median()
            features[f'{col}_skew'] = skew(series)
            features[f'{col}_kurt'] = kurtosis(series)
            features[f'{col}_q25'] = series.quantile(0.25)
            features[f'{col}_q75'] = series.quantile(0.75)
            features[f'{col}_iqr'] = series.quantile(0.75) - series.quantile(0.25)
    return features


def create_windows_and_features(df_group, cols_to_window_on): # Parameter renamed for clarity
    """Creates windows and calculates features for a group of data (one trial_id)."""
    all_window_features = []
    
    for i in range(0, len(df_group) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        window = df_group.iloc[i : i + WINDOW_SAMPLES]
        if len(window) < WINDOW_SAMPLES: 
            continue
            
        window_features = create_features_for_window(window, cols_to_window_on)
        window_features['trial_id'] = df_group['trial_id'].iloc[0]
        window_features['label'] = df_group['label'].iloc[0]
        all_window_features.append(window_features)
        
    return pd.DataFrame(all_window_features)


def process_all_data_to_features(full_raw_df):
    """Processes the complete raw DataFrame to extract windowed features."""
    if full_raw_df.empty:
        print("Input DataFrame is empty. No features will be generated.")
        return pd.DataFrame()

    print("Calculating magnitudes...")
    df_with_magnitudes, created_magnitude_columns = calculate_magnitudes(full_raw_df)
    
    # Define the complete list of columns to use in feature engineering
    # Includes the base sensor columns and the magnitude columns that were just created.
    cols_for_feature_engineering = BASE_SENSOR_COLUMNS[:] + [
        mc for mc in created_magnitude_columns if mc not in BASE_SENSOR_COLUMNS
    ]
    
    print(f"Columns used for feature engineering: {cols_for_feature_engineering}")

    print("Grouping by trial_id and generating window features...")
    all_trial_features = []
    grouped = df_with_magnitudes.groupby('trial_id')
    total_groups = len(grouped)
    current_group_idx = 0
    for trial_id, group_df in grouped:
        current_group_idx += 1
        print(f"Processing trial {current_group_idx}/{total_groups}: {trial_id}")
        # Pass the correct list of columns to the windowing function
        trial_features_df = create_windows_and_features(group_df, cols_for_feature_engineering)
        if not trial_features_df.empty:
            all_trial_features.append(trial_features_df)
            
    if not all_trial_features:
        print("No features were generated after windowing.")
        return pd.DataFrame()
        
    final_features_df = pd.concat(all_trial_features, ignore_index=True)
    print(f"Final features DataFrame with {len(final_features_df)} windows and {len(final_features_df.columns)} features.")
    if 'label' in final_features_df.columns:
        print("Label distribution in the features dataset:")
        print(final_features_df['label'].value_counts(normalize=True))
    return final_features_df

if __name__ == '__main__':
    print("This script is for feature engineering. To test, run main.py or integrate with data_loader.")
    try:
        from data_loader import load_all_data # R_ANKLE_COLUMNS, COLUMN_RENAMES are no longer needed here
        print("Loading raw data for feature engineering testing...")
        # Assuming data_loader.py is in the same directory or in the python path
        raw_data = load_all_data() 
        if not raw_data.empty:
            print("Processing data for features...")
            feature_data = process_all_data_to_features(raw_data)
            if not feature_data.empty:
                print("\nFirst 5 rows of the features dataset:")
                print(feature_data.head())
                print("\nFeatures dataset information:")
                feature_data.info()
                nan_counts = feature_data.isnull().sum()
                nan_features = nan_counts[nan_counts > 0]
                if not nan_features.empty:
                    print("\nFeatures with NaN values after engineering:")
                    print(nan_features)
                else:
                    print("\nNo NaN values found in the generated features.")
            else:
                print("Features dataset is empty.")
        else:
            print("Raw dataset is empty, couldn't generate features.")
    except ImportError:
        print("Error: data_loader.py not found. Run from the project root or adjust the import.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc() 