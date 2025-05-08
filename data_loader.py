import pandas as pd
import os
import glob

# Columns to be selected for right ankle
R_ANKLE_COLUMNS = [
    'Time',
    'r.ankle Acceleration X (m/s^2)',
    'r.ankle Acceleration Y (m/s^2)',
    'r.ankle Acceleration Z (m/s^2)',
    'r.ankle Angular Velocity X (rad/s)',
    'r.ankle Angular Velocity Y (rad/s)',
    'r.ankle Angular Velocity Z (rad/s)',
    'r.ankle Magnetic Field X (uT)',
    'r.ankle Magnetic Field Y (uT)',
    'r.ankle Magnetic Field Z (uT)'
]

# Mapping from column names to shorter and more friendly names
COLUMN_RENAMES = {
    'r.ankle Acceleration X (m/s^2)': 'r_acc_x',
    'r.ankle Acceleration Y (m/s^2)': 'r_acc_y',
    'r.ankle Acceleration Z (m/s^2)': 'r_acc_z',
    'r.ankle Angular Velocity X (rad/s)': 'r_gyro_x',
    'r.ankle Angular Velocity Y (rad/s)': 'r_gyro_y',
    'r.ankle Angular Velocity Z (rad/s)': 'r_gyro_z',
    'r.ankle Magnetic Field X (uT)': 'r_mag_x',
    'r.ankle Magnetic Field Y (uT)': 'r_mag_y',
    'r.ankle Magnetic Field Z (uT)': 'r_mag_z'
}

def find_xlsx_files(base_dir):
    """Finds all .xlsx files in the expected subdirectories."""
    patterns = [
        os.path.join(base_dir, 'sub*', 'ADLs', '*.xlsx'),
        os.path.join(base_dir, 'sub*', 'Falls', '*.xlsx'),
        os.path.join(base_dir, 'sub*', 'Near_Falls', '*.xlsx')
    ]
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    return all_files

def load_single_file(file_path, columns_to_keep, column_renames):
    """Loads a single .xlsx file, selects and renames columns."""
    try:
        df = pd.read_excel(file_path, usecols=columns_to_keep)
        df = df.rename(columns=column_renames)
        # Add trial_id and label
        parts = file_path.replace('\\', '/').split('/')
        subject_id = parts[-3] # e.g., sub1
        activity_type = parts[-2] # e.g., Falls
        file_name = parts[-1]
        
        df['trial_id'] = f"{subject_id}_{activity_type}_{file_name}"
        
        if activity_type == 'Falls':
            df['label'] = 1
        else: # ADLs and Near_Falls
            df['label'] = 0
            
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return pd.DataFrame() # Returns empty DataFrame in case of error

def load_all_data(base_dir='.'):
    """Loads all data, assigns labels and concatenates them."""
    all_files = find_xlsx_files(base_dir)
    if not all_files:
        print("No .xlsx files found. Check the base_dir and folder structure.")
        return pd.DataFrame()

    data_frames = []
    for file_path in all_files:
        print(f"Processing file: {file_path}")
        df_trial = load_single_file(file_path, R_ANKLE_COLUMNS, COLUMN_RENAMES)
        if not df_trial.empty:
            data_frames.append(df_trial)
    
    if not data_frames:
        print("No data was successfully loaded.")
        return pd.DataFrame()
        
    full_df = pd.concat(data_frames, ignore_index=True)
    print(f"Total of {len(all_files)} files processed.")
    print(f"Final DataFrame with {len(full_df)} rows and {len(full_df.columns)} columns.")
    print("Label distribution:")
    print(full_df['label'].value_counts(normalize=True))
    return full_df

if __name__ == '__main__':
    # Example of use (assuming the script is in the root of the AI_ML_Challenge project)
    print("Starting data loading...")
    dataset = load_all_data()
    
    if not dataset.empty:
        print("\nFirst 5 rows of the loaded dataset:")
        print(dataset.head())
        print("\nDataset information:")
        dataset.info()
    else:
        print("Dataset is empty after loading.") 