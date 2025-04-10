# Prediction-of-Fuel-Efficiency
Detailed Step-by-Step Explanation of the Auto MPG Prediction Project
This project uses deep learning (neural networks) to predict a car's fuel efficiency (MPG - Miles Per Gallon) based on various vehicle attributes. Below is a detailed breakdown of each step:

1. Data Loading & Initial Setup
Libraries Used
Pandas – Data manipulation & cleaning

NumPy – Numerical operations

Matplotlib & Seaborn – Data visualization

TensorFlow & Keras – Building & training the neural network

Dataset Overview
The dataset contains vehicle attributes:

MPG (Target Variable) – Fuel efficiency in miles per gallon.

Cylinders – Number of engine cylinders.

Displacement – Engine displacement (size).

Horsepower – Engine power.

Weight – Vehicle weight.

Acceleration – Time to reach 0-60 mph.

Model Year – Year of manufacture.

Origin – Country of origin (1: USA, 2: Europe, 3: Japan).

Loading the Data
The dataset is loaded from a raw string (for reproducibility).

Missing values (?) are treated as NaN for later cleaning.

2. Data Preprocessing
Handling Categorical Data (Origin)
The Origin column is one-hot encoded into three binary columns:

USA, Europe, Japan (1 if true, else 0).

Train-Test Split
80% training data, 20% test data (randomized).

Handling Missing Values
Rows with missing values (e.g., Horsepower = ?) are dropped.

Debugging checks ensure no NaN values remain.

Data Normalization
Features are standardized (mean=0, std=1) using:

Normalized
=
Value
−
Mean
Standard Deviation
+
ϵ
Normalized= 
Standard Deviation+ϵ
Value−Mean
​
 
(Small epsilon 1e-7 prevents division by zero.)

Why Normalize? Ensures all features contribute equally to training.

Saving Processed Data
Raw, cleaned, and normalized versions are saved as CSV for inspection.

3. Exploratory Data Analysis (EDA)
Pair Plots (sns.pairplot()) visualize relationships between:

MPG, Cylinders, Displacement, Weight.

Helps identify trends (e.g., higher weight → lower MPG).

4. Building the Neural Network
Model Architecture
A Sequential model with:

Input Layer: Shape matches number of features.

Two Hidden Layers (64 neurons each, ReLU activation).

Output Layer: Single neuron (predicts MPG).

Optimizer & Loss Function
Optimizer: RMSprop (learning rate = 0.001).

Loss Function: Mean Squared Error (MSE) (common for regression).

Metrics Tracked:

Mean Absolute Error (MAE) – Average error in MPG.

MSE – Squared error (penalizes larger errors).

5. Training the Model
Training Process
Epochs = 100 (iterations over the dataset).

Validation Split = 20% (monitor overfitting).

Early Stopping (patience=10):

Stops training if validation loss doesn’t improve for 10 epochs.

Visualizing Training Progress
Training vs. Validation Error Plots:

MAE & MSE over epochs.

Helps detect overfitting (if validation error rises).

6. Model Evaluation
Testing Performance
Mean Absolute Error (MAE): ~2.5 MPG (on test data).

Example: If true MPG = 30, prediction is likely 27.5–32.5.

Prediction Visualization
Scatter Plot: True MPG vs. Predicted MPG.

Ideal Line (y=x): Shows how close predictions are to reality.

7. Key Insights & Improvements
Findings
Weight & Displacement strongly affect MPG (negative correlation).

More cylinders → Lower MPG (due to higher fuel consumption).

Possible Improvements
Feature Engineering (e.g., Weight/HP ratio).

Hyperparameter Tuning (adjust layers, neurons, learning rate).

Cross-Validation (better generalization).

Different Models (Random Forest, XGBoost for comparison).

Conclusion
This project demonstrates:
✅ End-to-end ML pipeline (data cleaning → training → evaluation).
✅ Neural Networks for regression (predicting continuous values).
✅ Best practices (normalization, early stopping, EDA).

Use Case:

Automotive industry (fuel efficiency optimization).

Environmental impact studies.

Would you like any modifications or additional experim
