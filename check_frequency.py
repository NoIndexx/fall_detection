import pandas as pd

try:
    # Path to the file (adjust as needed if the script is not in the root)
    file_path = 'sub1/Falls/AXR_ITCS_trial1.xlsx'
    df = pd.read_excel(file_path, nrows=10) # Read the first 10 lines
    
    if 'Time' in df.columns:
        time_diffs = df['Time'].diff().dropna()
        print("Time differences (microseconds):")
        print(time_diffs)
        
        if not time_diffs.empty:
            mean_diff_us = time_diffs.mean()
            print(f"\nMean time difference: {mean_diff_us} us")
            frequency = 1 / (mean_diff_us / 1_000_000) # Converting µs to s
            print(f"Estimated sampling frequency: {frequency:.2f} Hz")
        else:
            print("Could not calculate frequency (few samples or 'Time' column not numeric in the first lines).")
            if not df['Time'].empty:
                 print("First values of the 'Time' column:")
                 print(df['Time'])

    else:
        print("Column 'Time' not found in file.")
        print("Available columns:", df.columns.tolist())

except Exception as e:
    print(f"An error occurred: {e}") 