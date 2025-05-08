import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import os
from unittest.mock import patch, mock_open

# Import functions to be tested
from data_loader import find_xlsx_files, load_single_file, load_all_data, R_ANKLE_COLUMNS, COLUMN_RENAMES

@pytest.fixture
def sample_raw_df_content():
    # Provides a minimal DataFrame structure that load_single_file expects
    # after pd.read_excel, before renaming and adding label/trial_id
    data = {
        'Time': [1000, 2000, 3000],
        'r.ankle Acceleration X (m/s^2)': [1, 2, 3],
        'r.ankle Acceleration Y (m/s^2)': [4, 5, 6],
        'r.ankle Acceleration Z (m/s^2)': [7, 8, 9],
        'r.ankle Angular Velocity X (rad/s)': [0.1, 0.2, 0.3],
        'r.ankle Angular Velocity Y (rad/s)': [0.4, 0.5, 0.6],
        'r.ankle Angular Velocity Z (rad/s)': [0.7, 0.8, 0.9],
        'r.ankle Magnetic Field X (uT)': [10, 20, 30],
        'r.ankle Magnetic Field Y (uT)': [40, 50, 60],
        'r.ankle Magnetic Field Z (uT)': [70, 80, 90]
    }
    return pd.DataFrame(data)

@pytest.fixture
def expected_renamed_df_structure():
    # Expected structure after renaming, before label/trial_id
    # This helps test COLUMN_RENAMES and selection logic
    data = {
        'Time': [1000, 2000, 3000],
        'r_acc_x': [1, 2, 3],
        'r_acc_y': [4, 5, 6],
        'r_acc_z': [7, 8, 9],
        'r_gyro_x': [0.1, 0.2, 0.3],
        'r_gyro_y': [0.4, 0.5, 0.6],
        'r_gyro_z': [0.7, 0.8, 0.9],
        'r_mag_x': [10, 20, 30],
        'r_mag_y': [40, 50, 60],
        'r_mag_z': [70, 80, 90]
    }
    return pd.DataFrame(data)


@patch('data_loader.glob.glob')
def test_find_xlsx_files(mock_glob):
    mock_glob.side_effect = [
        ['./sub1/ADLs/file1.xlsx'], # Mock response for ADLs pattern
        ['./sub1/Falls/file2.xlsx'], # Mock response for Falls pattern
        ['./sub1/Near_Falls/file3.xlsx'] # Mock response for Near_Falls pattern
    ]
    base_dir = '.'
    expected_files = ['./sub1/ADLs/file1.xlsx', './sub1/Falls/file2.xlsx', './sub1/Near_Falls/file3.xlsx']
    actual_files = find_xlsx_files(base_dir)
    assert actual_files == expected_files
    assert mock_glob.call_count == 3 # Ensure it tried all patterns

@patch('data_loader.pd.read_excel')
def test_load_single_file_fall(mock_read_excel, sample_raw_df_content):
    mock_read_excel.return_value = sample_raw_df_content.copy() # Use the fixture
    
    file_path = "./sub1/Falls/AXR_ITCS_trial1.xlsx" # Example path
    df_loaded = load_single_file(file_path, R_ANKLE_COLUMNS, COLUMN_RENAMES)
    
    mock_read_excel.assert_called_once_with(file_path, usecols=R_ANKLE_COLUMNS)
    
    assert 'trial_id' in df_loaded.columns
    assert 'label' in df_loaded.columns
    assert df_loaded['label'].iloc[0] == 1 # Falls should be 1
    assert df_loaded['trial_id'].iloc[0] == "sub1_Falls_AXR_ITCS_trial1.xlsx"
    
    # Check if renaming worked as expected
    expected_cols = list(COLUMN_RENAMES.values()) + ['Time', 'trial_id', 'label']
    # Sort for comparison as column order might vary slightly after operations
    assert sorted(list(df_loaded.columns)) == sorted(expected_cols)


@patch('data_loader.pd.read_excel')
def test_load_single_file_adl(mock_read_excel, sample_raw_df_content):
    mock_read_excel.return_value = sample_raw_df_content.copy()
    
    file_path = "./sub2/ADLs/TXI_AS_trial1.xlsx"
    df_loaded = load_single_file(file_path, R_ANKLE_COLUMNS, COLUMN_RENAMES)
    
    assert df_loaded['label'].iloc[0] == 0 # ADLs should be 0
    assert df_loaded['trial_id'].iloc[0] == "sub2_ADLs_TXI_AS_trial1.xlsx"

@patch('data_loader.find_xlsx_files')
@patch('data_loader.load_single_file')
def test_load_all_data(mock_load_single, mock_find_files, sample_raw_df_content, expected_renamed_df_structure):
    # Mock find_xlsx_files to return a predefined list of dummy file paths
    mock_find_files.return_value = ['file1.xlsx', 'file2.xlsx']
    
    # Mock load_single_file to return a consistent DataFrame structure
    # We'll add 'trial_id' and 'label' as load_single_file would
    df_file1 = expected_renamed_df_structure.copy()
    df_file1['trial_id'] = 'file1'
    df_file1['label'] = 0
    
    df_file2 = expected_renamed_df_structure.copy()
    df_file2['trial_id'] = 'file2'
    df_file2['label'] = 1
        
    mock_load_single.side_effect = [df_file1, df_file2]

    combined_df = load_all_data(base_dir='dummy_dir')
    
    mock_find_files.assert_called_once_with('dummy_dir')
    assert mock_load_single.call_count == 2
    
    # Expected DataFrame is concatenation of df_file1 and df_file2
    expected_df = pd.concat([df_file1, df_file2], ignore_index=True)
    
    assert_frame_equal(combined_df, expected_df)
    assert len(combined_df) == len(sample_raw_df_content) * 2 # 2 files, each 3 rows from fixture
    assert 'trial_id' in combined_df.columns
    assert 'label' in combined_df.columns

def test_load_all_data_no_files_found():
    with patch('data_loader.find_xlsx_files', return_value=[]):
        df = load_all_data(base_dir='empty_dir')
        assert df.empty

def test_load_single_file_read_error():
    with patch('data_loader.pd.read_excel', side_effect=Exception("Read error")):
        df = load_single_file("bad_file.xlsx", R_ANKLE_COLUMNS, COLUMN_RENAMES)
        assert df.empty 