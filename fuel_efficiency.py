import io
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
data = """
18.0   8   307.0      130.0      3504.      12.0   70  1
15.0   8   350.0      165.0      3693.      11.5   70  1
18.0   8   318.0      150.0      3436.      11.0   70  1
16.0   8   304.0      150.0      3433.      12.0   70  1
17.0   8   302.0      140.0      3449.      10.5   70  1
15.0   8   429.0      198.0      4341.      10.0   70  1
14.0   8   454.0      220.0      4354.       9.0   70  1
14.0   8   440.0      215.0      4312.       8.5   70  1
14.0   8   455.0      225.0      4425.      10.0   70  1
15.0   8   390.0      190.0      3850.       8.5   70  1
15.0   8   383.0      170.0      3563.      10.0   70  1
14.0   8   340.0      160.0      3609.       8.0   70  1
15.0   8   400.0      150.0      3761.       9.5   70  1
14.0   8   455.0      225.0      3086.      10.0   70  1
24.0   4   113.0       95.00     2372.      15.0   70  3
22.0   4   98.00       ?         2130.      14.5   70  2
18.0   6   199.0      97.00      2774.      15.5   70  1
21.0   6   200.0      85.00      2587.      16.0   70  1
27.0   4   97.00      88.00      2130.      14.5   70  3
26.0   4   97.00      46.00      1835.      20.5   70  2
25.0   4   110.0      87.00      2672.      17.5   70  2
24.0   4   107.0      90.00      2430.      14.5   70  2
25.0   4   104.0      95.00      2375.      17.5   70  2
26.0   4   121.0      113.       2234.      12.5   70  2
21.0   6   199.0      90.00      2648.      15.0   70  1
10.0   8   360.0      215.0      4615.      14.0   70  1
10.0   8   307.0      200.0      4376.      15.0   70  1
11.0   8   318.0      210.0      4382.      13.5   70  1
9.0    8   304.0      193.0      4732.      18.5   70  1
27.0   4   97.00      88.00      2130.      14.5   71  3
28.0   4   140.0      90.00      2264.      15.5   71  1
25.0   4   113.0      95.00      2228.      14.0   71  3
25.0   4   98.00       ?         2046.      19.0   71  1
19.0   6   232.0      100.0      2634.      13.0   71  1
16.0   6   225.0      105.0      3439.      15.5   71  1
17.0   6   250.0      100.0      3329.      15.5   71  1
19.0   6   250.0      88.00      3302.      15.5   71  1
18.0   6   232.0      100.0      3288.      15.5   71  1
14.0   8   350.0      165.0      4209.      12.0   71  1
14.0   8   400.0      175.0      4464.      11.5   71  1
14.0   8   351.0      153.0      4154.      13.5   71  1
14.0   8   318.0      150.0      4096.      13.0   71  1
12.0   8   383.0      180.0      4955.      11.5   71  1
13.0   8   400.0      170.0      4746.      12.0   71  1
13.0   8   400.0      175.0      5140.      12.0   71  1
18.0   6   258.0      110.0      2962.      13.5   71  1
22.0   4   140.0      72.00      2408.      19.0   71  1
19.0   6   250.0      100.0      3282.      15.0   71  1
18.0   6   250.0      88.00      3139.      14.5   71  1
23.0   4   122.0      86.00      2220.      14.0   71  1
28.0   4   116.0      90.00      2123.      14.0   71  2
30.0   4   79.00      70.00      2074.      19.5   71  2
30.0   4   88.00      76.00      2065.      14.5   71  2
31.0   4   71.00      65.00      1773.      19.0   71  3
35.0   4   72.00      69.00      1613.      18.0   71  3
27.0   4   97.00      60.00      1834.      19.0   71  2
26.0   4   91.00      70.00      1955.      20.5   71  1
24.0   4   113.0      95.00      2278.      15.5   72  3
25.0   4   97.50      80.00      2126.      17.0   72  1
23.0   4   97.00      54.00      2254.      23.5   72  2
20.0   4   140.0      90.00      2408.      19.5   72  1
15.0   8   400.0      190.0      4325.      12.2   72  1
22.0   4   108.0      94.00      2379.      16.5   72  3
18.0   6   225.0      105.0      3613.      16.0   72  1
21.0   6   231.0      110.0      3900.      21.0   72  1
27.0   4   140.0      75.00      2155.      14.4   72  1
26.0   4   98.00      79.00      2255.      17.7   72  1
25.0   4   134.0      96.00      2511.      14.8   72  3
24.0   4   119.0      97.00      2545.      15.0   72  3
25.0   4   105.0      75.00      2213.      14.4   72  1
26.0   4   134.0      95.00      2515.      14.8   72  3
24.0   4   120.0      97.00      2500.      14.9   72  3
22.0   4   121.0      98.00      2945.      14.5   72  2
23.0   4   121.0      115.       2671.      13.5   72  2
16.0   6   250.0      100.0      3781.      17.0   72  1
19.0   6   250.0      88.00      3021.      16.5   72  1
18.0   6   232.0      100.0      2901.      16.0   72  1
24.0   4   115.0      95.00      2694.      15.0   72  3
22.0   4   120.0      97.00      2489.      15.0   72  3
29.0   4   97.00      71.00      1825.      12.2   72  2
27.0   4   140.0      86.00      2790.      15.6   72  1
23.0   4   113.0      95.00      2268.      15.5   72  3
24.0   4   120.0       ?         2615.      16.8   72  3
25.0   4   121.0      112.       2868.      15.5   72  2
26.0   4   96.00      69.00      2189.      18.0   72  2
23.0   4   122.0      86.00      2395.      16.0   72  1
20.0   4   156.0      92.00      2625.      14.4   72  1
24.0   4   120.0      74.00      2635.      18.3   72  1
22.0   4   140.0      72.00      2401.      19.5   72  1
28.0   4   107.0      90.00      2430.      14.5   72  2
19.0   6   225.0      95.00      3410.      16.6   72  1
29.0   4   98.00      83.00      2219.      16.5   72  2
23.0   4   151.0       ?         2556.      13.2   72  1
22.0   6   200.0       ?         2870.      17.0   72  1
24.0   4   140.0      92.00      2865.      16.4   72  1
23.0   4   90.00      70.00      1937.      14.2   72  2
44.0   4   97.00      52.00      2130.      24.6   72  2
32.0   4   135.0      84.00      2295.      11.6   72  1
28.0   4   120.0      79.00      2625.      18.6   72  1
31.0   4   119.0      82.00      2720.      19.4   72  1
"""
dataset = pd.read_csv(io.StringIO(data), names=column_names,
                     na_values="?", comment='\t',
                     sep="\s+", skipinitialspace=True)

# One-hot encode the origin column
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0

# Split into train and test sets
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Visualize the data
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

# Separate labels from features
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Validate and normalize the data
try:
    print("\nData validation:")
    print("Missing values in training set:", train_dataset.isnull().sum().sum())
    print("Missing values in test set:", test_dataset.isnull().sum().sum())
    
    # Clean data by dropping rows with missing values
    print("\nCleaning data by dropping rows with missing values...")
    train_dataset = train_dataset.dropna()
    test_dataset = test_dataset.dropna()
    train_labels = train_labels[train_dataset.index]
    test_labels = test_labels[test_dataset.index]
    
    print("Training set size after cleaning:", len(train_dataset))
    print("Test set size after cleaning:", len(test_dataset))
    
    # Save raw data for inspection with debug prints
    print("\nSaving raw data files...")
    train_path = 'train_data_raw.csv'
    test_path = 'test_data_raw.csv'
    print(f"Writing training data to: {os.path.abspath(train_path)}")
    train_dataset.to_csv(train_path)
    print(f"Writing test data to: {os.path.abspath(test_path)}")
    test_dataset.to_csv(test_path)
    print("Files saved successfully")

    train_stats = train_dataset.describe().transpose()
    print("\nTraining stats:")
    print(train_stats)
    
    # Save stats for inspection with debug prints
    stats_path = 'train_stats.csv'
    print(f"\nWriting stats to: {os.path.abspath(stats_path)}")
    train_stats.to_csv(stats_path)
    print("Stats file saved successfully")

    def norm(x):
        # Add small epsilon to avoid division by zero
        normalized = (x - train_stats['mean']) / (train_stats['std'] + 1e-7)
        if normalized.isnull().values.any():
            raise ValueError("NaN values detected after normalization - this should not happen after cleaning")
        return normalized
    
    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    
    # Save normalized data with debug prints
    norm_train_path = 'train_data_normalized.csv'
    norm_test_path = 'test_data_normalized.csv'
    print(f"\nWriting normalized training data to: {os.path.abspath(norm_train_path)}")
    normed_train_data.to_csv(norm_train_path)
    print(f"Writing normalized test data to: {os.path.abspath(norm_test_path)}")
    normed_test_data.to_csv(norm_test_path)
    print("Normalized data files saved successfully")

    print("\nNormalized data samples:")
    print(normed_train_data.head())
    
except Exception as e:
    print(f"\nERROR during data processing: {str(e)}")
    raise

# Build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model
model = build_model()

# Train the model
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 100

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

# Visualize training
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)

# Early stopping
model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
plot_history(history)

# Evaluate on test set
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# Make predictions
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()
