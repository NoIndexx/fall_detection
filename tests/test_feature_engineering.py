import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

from feature_engineering import (
    calculate_magnitudes, 
    create_features_for_window, 
    create_windows_and_features,
    process_all_data_to_features,
    BASE_SENSOR_COLUMNS,
    SAMPLING_FREQUENCY,
    WINDOW_SAMPLES,
    STEP_SAMPLES
)

@pytest.fixture
def sample_sensor_data():
    data = {
        'Time': range(WINDOW_SAMPLES * 2), # Enough data for a few windows
        'r_acc_x': np.random.rand(WINDOW_SAMPLES * 2),
        'r_acc_y': np.random.rand(WINDOW_SAMPLES * 2),
        'r_acc_z': np.random.rand(WINDOW_SAMPLES * 2),
        'r_gyro_x': np.random.rand(WINDOW_SAMPLES * 2),
        'r_gyro_y': np.random.rand(WINDOW_SAMPLES * 2),
        'r_gyro_z': np.random.rand(WINDOW_SAMPLES * 2),
        'r_mag_x': np.random.rand(WINDOW_SAMPLES * 2),
        'r_mag_y': np.random.rand(WINDOW_SAMPLES * 2),
        'r_mag_z': np.random.rand(WINDOW_SAMPLES * 2),
        'trial_id': ['trial1'] * (WINDOW_SAMPLES * 2),
        'label': [0] * (WINDOW_SAMPLES * 2)
    }
    return pd.DataFrame(data)

def test_calculate_magnitudes(sample_sensor_data):
    df_input = sample_sensor_data[BASE_SENSOR_COLUMNS].copy()
    df_output, created_mag_cols = calculate_magnitudes(df_input)
    
    assert 'r_acc_mag' in df_output.columns
    assert 'r_gyro_mag' in df_output.columns
    assert 'r_mag_mag' in df_output.columns
    assert 'r_acc_mag' in created_mag_cols
    assert 'r_gyro_mag' in created_mag_cols
    assert 'r_mag_mag' in created_mag_cols
    
    expected_acc_mag = np.sqrt(df_input['r_acc_x']**2 + df_input['r_acc_y']**2 + df_input['r_acc_z']**2)
    expected_acc_mag.name = 'r_acc_mag' # Assign name to the expected series
    assert_series_equal(df_output['r_acc_mag'], expected_acc_mag, check_dtype=False)

def test_create_features_for_window(sample_sensor_data):
    # Test with a single window of data
    window_df = sample_sensor_data.iloc[:WINDOW_SAMPLES]
    # Get all columns that would have magnitudes calculated + base sensors
    cols_to_feature = BASE_SENSOR_COLUMNS + ['r_acc_mag', 'r_gyro_mag', 'r_mag_mag']
    
    # Manually add magnitude columns for this test window, as calculate_magnitudes is tested separately
    window_df_with_mag = window_df.copy()
    window_df_with_mag['r_acc_mag'] = np.sqrt(window_df['r_acc_x']**2 + window_df['r_acc_y']**2 + window_df['r_acc_z']**2)
    window_df_with_mag['r_gyro_mag'] = np.sqrt(window_df['r_gyro_x']**2 + window_df['r_gyro_y']**2 + window_df['r_gyro_z']**2)
    window_df_with_mag['r_mag_mag'] = np.sqrt(window_df['r_mag_x']**2 + window_df['r_mag_y']**2 + window_df['r_mag_z']**2)

    features = create_features_for_window(window_df_with_mag, cols_to_feature)
    
    # Expect 11 features (mean, std, min, max, range, median, skew, kurt, q25, q75, iqr) for each of the 12 columns
    # 9 base sensor cols + 3 magnitude cols = 12 cols for features
    # So, 12 cols * 11 stats/col = 132 features
    assert len(features) == 12 * 11
    assert f'{BASE_SENSOR_COLUMNS[0]}_mean' in features
    assert f'r_acc_mag_std' in features

def test_create_windows_and_features(sample_sensor_data):
    df_group = sample_sensor_data.copy()
    # Simulate having magnitudes already calculated
    df_group_with_mag, mag_cols = calculate_magnitudes(df_group)
    cols_to_window = BASE_SENSOR_COLUMNS + mag_cols

    windowed_features_df = create_windows_and_features(df_group_with_mag, cols_to_window)
    
    # With WINDOW_SAMPLES = 256, STEP_SAMPLES = 128, and len(df_group) = 512 (WINDOW_SAMPLES * 2)
    # Number of windows = floor((total_samples - window_size) / step_size) + 1
    # = floor((512 - 256) / 128) + 1 = floor(256 / 128) + 1 = 2 + 1 = 3 windows
    expected_num_windows = np.floor((len(df_group) - WINDOW_SAMPLES) / STEP_SAMPLES) + 1
    assert len(windowed_features_df) == expected_num_windows
    assert 'trial_id' in windowed_features_df.columns
    assert 'label' in windowed_features_df.columns
    assert f'{BASE_SENSOR_COLUMNS[0]}_mean' in windowed_features_df.columns 
    assert f'r_mag_mag_iqr' in windowed_features_df.columns

def test_process_all_data_to_features_empty_input():
    empty_df = pd.DataFrame()
    processed_df = process_all_data_to_features(empty_df)
    assert processed_df.empty

def test_process_all_data_to_features_normal_case(sample_sensor_data):
    # This is a more integrative test
    # It uses the sample_sensor_data which has one trial_id
    processed_df = process_all_data_to_features(sample_sensor_data)
    
    expected_num_windows = np.floor((len(sample_sensor_data) - WINDOW_SAMPLES) / STEP_SAMPLES) + 1
    assert not processed_df.empty
    assert len(processed_df) == expected_num_windows
    assert 'label' in processed_df.columns
    assert 'trial_id' in processed_df.columns
    # Check a few expected feature columns (12 cols * 11 stats = 132 features + trial_id + label = 134 cols)
    assert len(processed_df.columns) == (len(BASE_SENSOR_COLUMNS) + 3) * 11 + 2 # +3 for 3 mag cols, +2 for trial_id, label
    assert 'r_acc_x_mean' in processed_df.columns
    assert 'r_mag_mag_kurt' in processed_df.columns 