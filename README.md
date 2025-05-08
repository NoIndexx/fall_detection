# Fall Detection System Using IMU Sensor Data

**Developed by: Renato Cesar de Oliveira**

**Repository: [https://github.com/NoIndexx/fall_detection.git](https://github.com/NoIndexx/fall_detection.git)**

## Project Overview

This project implements a machine learning system for detecting falls in elderly individuals using Inertial Measurement Unit (IMU) sensor data. Falls are a significant health risk for elderly populations, often resulting in serious injuries and decreased quality of life. Early detection of falls can significantly improve emergency response times and health outcomes.

The system uses data from body-worn sensors placed at multiple locations, with a focus on right ankle measurements collected at **128Hz**, to classify movements as either falls or non-falls (Activities of Daily Living - ADLs or Near Falls).

## Features

- Processes time-series data from IMU sensors (accelerometer, gyroscope, magnetometer)
- Uses windowing techniques to extract meaningful features
- Implements a Random Forest classifier for fall detection
- Achieves high accuracy and precision in fall detection
- Provides comprehensive evaluation metrics and visualizations (**saved to `plots/` directory**)
- **Saves processed features to a CSV file (`processed_features_right_ankle.csv`)**

> **Note:** The plots in the `plots/` directory and the `processed_features_right_ankle.csv` file are included for reference only. While results should be reproducible due to the fixed random seed (42), it's recommended to run the code yourself for verification or if you want to modify parameters.

## Data Description

The dataset contains IMU sensor data collected from 8 subjects (healthy young adults with ages ranging from 22 to 32 years) across three types of activities:

1. **Activities of Daily Living (ADLs)** - Normal daily activities like walking, sitting, standing, etc.
2. **Falls** - Various types of falls, including forward, backward, and sideways falls
3. **Near Falls** - Situations where a fall was imminent but was prevented

Each subject wore IMU sensors at 7 different body locations:
- Right ankle
- Left ankle
- Right thigh
- Left thigh
- Head
- Sternum
- Waist

For each location, the following measurements were recorded at **~128Hz**:
- 3-axis acceleration (m/s²)
- 3-axis angular velocity (rad/s)
- 3-axis magnetic field (μT)

## Setup and Installation

### Prerequisites

- Python 3.8+ (Project developed with Python 3.8.10)
- Pip package manager

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/NoIndexx/fall_detection.git
   cd fall-detection
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download and organize the dataset (already included):
   - The dataset should be organized with the following folder structure:
     ```
     AI_ML_Challenge/
     ├── sub1/
     │   ├── ADLs/
     │   ├── Falls/
     │   └── Near_Falls/
     ├── sub2/
     │   ├── ADLs/
     │   ├── Falls/
     │   └── Near_Falls/
     ├── ...
     └── sub8/
         ├── ADLs/
         ├── Falls/
         └── Near_Falls/
     ```

## Usage

To run the full pipeline (data loading, feature engineering, model training, and evaluation):

```
python main.py
```

The script will:
1. Load raw data from the dataset (`sub*/*/*.xlsx` files)
2. Process and extract features from the right ankle sensor data
3. **Save the processed features to `processed_features_right_ankle.csv`**
4. Train a Random Forest classifier
5. Evaluate the model performance on a test set
6. Print evaluation metrics to the console
7. Generate and save evaluation plots (confusion matrix, ROC curve) in the `plots/` directory

### Reproducibility

For reproducibility purposes, the model uses a fixed random seed (42) for both data splitting and Random Forest training. This ensures that running the code will produce consistent results across different executions. If you need to modify this seed, you can find it in the `model.py` file.

## Testing

This project uses `pytest` for unit testing. Tests are located in the `tests/` directory.

To run the tests:

1.  Ensure you have `pytest` installed (it's included in `requirements.txt`).
2.  Navigate to the project root directory in your terminal.
3.  Run the following command:

    ```bash
    python -m pytest tests -v
    ```

This will automatically discover and run all tests in the `tests/` directory with verbose output. The tests cover:
- Data loading (`tests/test_data_loader.py`)
- Feature engineering (`tests/test_feature_engineering.py`)
- Model training and prediction (`tests/test_model.py`)

## Methodology

### Why Right Ankle Sensor?

Based on research in fall detection literature (including the referenced article by Paolini et al.), sensors placed at the shinbone/ankle area have been shown to provide higher accuracy for fall detection compared to other body locations. This is likely because:

1. The lower extremities experience significant dynamic changes during falls
2. Ankle motion patterns differ distinctly between falls and ADLs
3. The ankle location provides clear signal patterns with less noise than sensors at other locations

### Data Processing

The data processing pipeline consists of the following steps:

1. **Data Loading**:
   - Load data from all subjects and trials
   - Extract the right ankle sensor data
   - Label falls as 1, ADLs and Near Falls as 0

2. **Feature Engineering**:
   - Calculate vector magnitudes for acceleration, angular velocity, and magnetic field
   - Apply a sliding window approach (**2-second windows with 1-second overlap, 128Hz sampling rate**)
   - Extract statistical features for each window **for the 9 original axes and 3 magnitude signals**, including:
     - Mean, standard deviation, minimum, maximum, range
     - Median, skewness, kurtosis
     - Quartiles (25th, 75th) and interquartile range (IQR)

3. **Data Preprocessing**:
   - Split the **windowed feature data** into training (80%) and testing (20%) sets, stratified by label
   - Standardize features using `StandardScaler` (fit on training data, applied to both train and test)
   - Handle any potential missing values by filling with column means (though none were observed in the initial run)

### Model Selection

The **Random Forest classifier** was selected for several reasons:

1. **Robustness to outliers and noise**: IMU data can be noisy, and Random Forests are less affected by outliers.
2. **Effective with high-dimensional data**: The feature engineering process produces many features, which Random Forests handle well.
3. **Handles non-linear relationships**: The relationship between sensor data and falls is complex and non-linear.
4. **Class imbalance management**: Random Forests can be configured to handle class imbalance through class weights.
5. **Feature importance**: Random Forests provide insights into feature importance, helpful for understanding which measurements are most indicative of falls.
6. **Less sensitive to multicollinearity**: Unlike LSTM and temporal regression models, Random Forests are more robust to multicollinearity among features, which is common in IMU sensor data where different axes may capture related movements.

The model was configured with **150 trees** and **balanced class weights** (`class_weight='balanced'`) to account for the slight class imbalance (approximately 66% non-falls, 34% falls).

## Results

### Performance Metrics

The initial baseline model achieved the following performance metrics on the test set:

- **Accuracy**: 89.14%
- **Precision (for Falls)**: 91.86%
- **Recall (for Falls)**: 74.47%
- **F1-Score (for Falls)**: 82.26%
- **ROC AUC**: 96.98%

### Confusion Matrix

(Refer to `plots/Random_Forest_Baseline_confusion_matrix.png` generated by `main.py`)

The confusion matrix shows:
- Excellent detection of non-falls (**~97%** recall for non-falls)
- Good detection of falls (**~74%** recall for falls)
- High precision for falls (**~92%**), indicating few false alarms.

### Interpretation

The model performs well as a baseline, showing strong discriminatory power (high AUC). The primary area for potential improvement is the **recall for falls**, meaning reducing the number of actual falls missed by the system. The high precision is beneficial for avoiding unnecessary alerts.

## Findings and Conclusions

1. **Right ankle sensor data is sufficient**: Using only the right ankle sensor data, the model achieved high accuracy, supporting the hypothesis that this location is optimal for fall detection.

2. **Statistical features capture fall dynamics**: The statistical features extracted from 2-second windows effectively capture the motion patterns that differentiate falls from normal activities.

3. **Random Forest provides robust classification**: The Random Forest classifier effectively learns the complex patterns in the data and provides good generalization to unseen examples.

4. **Trade-off between precision and recall**: The model achieves extremely high precision at the expense of some recall. Depending on the application, this balance could be adjusted.

5. **Real-time implementation potential**: The feature engineering and classification approach is computationally efficient and could be implemented in real-time fall detection systems.

## Future Work

1. **Incorporate sensor fusion**: Integrate data from multiple body locations to potentially improve detection accuracy.

2. **Feature selection**: Apply feature selection techniques to identify the most important features and reduce dimensionality.

3. **Deep learning approaches**: Explore recurrent neural networks (RNNs) or convolutional neural networks (CNNs) for automatic feature extraction from raw sensor data.

4. **Subject-specific models**: Investigate personalized models that adapt to individual gait and movement patterns.

5. **Real-time implementation**: Develop an embedded system for real-time fall detection using the developed algorithm.

6. **Hyperparameter optimization**: Further tune model parameters to improve performance, particularly recall.

## Project Structure

```
AI_ML_Challenge/
├── sub1/                     # Subject 1 raw data
│   ├── ADLs/
│   ├── Falls/
│   └── Near_Falls/
├── sub2/ ...                 # Other subjects
├── tests/                    # Unit tests
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   └── test_model.py
├── data_loader.py          # Loads and consolidates raw data
├── feature_engineering.py  # Extracts windowed features
├── model.py                # Splits data, trains Random Forest
├── evaluation.py           # Calculates metrics, saves plots
├── main.py                 # Main script orchestrating the pipeline
├── check_frequency.py      # Utility script (can be removed if desired)
├── plots/                    # Directory created for output plots
│   ├── Random_Forest_Baseline_confusion_matrix.png
│   └── Random_Forest_Baseline_roc_curve.png
├── processed_features_right_ankle.csv # Output CSV with processed features
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── data_description.txt    # Original data description
└── fall detection challenge.txt # Original challenge description
```

## License

**Proprietary Software - All Rights Reserved**

© 2024 Renato Cesar de Oliveira

This software and associated documentation files (the "Software") is proprietary and confidential. 
All rights are reserved.

**Restrictions:**
- You may not use this Software for any commercial purposes without explicit written permission.
- You may not distribute, sublicense, or transfer the Software to any third party.
- You may not modify, adapt, or create derivative works based on the Software.
- You may not reverse engineer, decompile, or disassemble the Software.

Unauthorized use, reproduction, or distribution of this Software may result in severe civil and criminal penalties, and will be prosecuted to the maximum extent possible under law.

## Acknowledgements

- Data was collected as part of a study on fall detection in elderly populations
- Special thanks to the researchers and participants who contributed to the dataset 