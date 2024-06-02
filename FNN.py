import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns

# Seaborn theme setup for two-column LaTeX document
sns.set_theme(style="whitegrid", palette="colorblind")
cm = 1/2.54  # conversion factor for cm to inches for plot sizing

# Define plot parameters suitable for LaTeX integration
plt.rcParams.update({
    "legend.fontsize": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "font.size": 7,
    "figure.figsize": (9 * cm, 12 * cm),
    "lines.markersize": 2.0,
    "lines.linewidth": 0.5,
    "grid.linestyle": '--',
    "grid.alpha": 0.6
})

# Seed setting for reproducibility
np.random.seed(2024)
tf.random.set_seed(2024)

# Load and preprocess the dataset
data = pd.read_csv("Data/processed_iskoras_measurements.csv")
feature_cols = ["co2_flux_filtered", "ch4_flux_filtered"]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[feature_cols].values)

# Split the data into training and testing sets
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]
X_train, y_train = train_data[:-1], train_data[1:]
X_test, y_test = test_data[:-1], test_data[1:]

class FNNRegression:
    """
    Feedforward Neural Network for Regression.
    Attributes:
        units (list): List of units in each layer.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        verbose (int): Verbosity mode.
    """
    def __init__(self, units=[100], num_epochs=1000, learning_rate=0.001, verbose=1):
        self.units = [units] if isinstance(units, int) else units
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, X_train, y_train):
        """
        Train the FNN model.
        Args:
            X_train (array): Training features.
            y_train (array): Training targets.
        """
        model = Sequential([
            Dense(self.units[0], activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(2)  # Output layer for both CO2 and CH4
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose)
        self.model = model

    def predict(self, X_test):
        """
        Predict using the FNN model.
        Args:
            X_test (array): Testing features.
        Returns:
            array: Predicted values.
        """
        return self.model.predict(X_test)

# Create and train the model
fnn_model = FNNRegression()
fnn_model.fit(X_train, y_train)
train_predictions = fnn_model.predict(X_train)
test_predictions = fnn_model.predict(X_test)

# Inverse transform to original scale
train_predictions_inverse = scaler.inverse_transform(train_predictions)
test_predictions_inverse = scaler.inverse_transform(test_predictions)
y_train_inverse = scaler.inverse_transform(y_train)
y_test_inverse = scaler.inverse_transform(y_test)

# Calculate metrics for training set
train_mse_co2 = mean_squared_error(y_train_inverse[:, 0], train_predictions_inverse[:, 0])
train_mae_co2 = mean_absolute_error(y_train_inverse[:, 0], train_predictions_inverse[:, 0])
train_rmse_co2 = np.sqrt(train_mse_co2)
train_r2_co2 = r2_score(y_train_inverse[:, 0], train_predictions_inverse[:, 0])

train_mse_ch4 = mean_squared_error(y_train_inverse[:, 1], train_predictions_inverse[:, 1])
train_mae_ch4 = mean_absolute_error(y_train_inverse[:, 1], train_predictions_inverse[:, 1])
train_rmse_ch4 = np.sqrt(train_mse_ch4)
train_r2_ch4 = r2_score(y_train_inverse[:, 1], train_predictions_inverse[:, 1])

# Calculate metrics for test set
test_mse_co2 = mean_squared_error(y_test_inverse[:, 0], test_predictions_inverse[:, 0])
test_mae_co2 = mean_absolute_error(y_test_inverse[:, 0], test_predictions_inverse[:, 0])
test_rmse_co2 = np.sqrt(test_mse_co2)
test_r2_co2 = r2_score(y_test_inverse[:, 0], test_predictions_inverse[:, 0])

test_mse_ch4 = mean_squared_error(y_test_inverse[:, 1], test_predictions_inverse[:, 1])
test_mae_ch4 = mean_absolute_error(y_test_inverse[:, 1], test_predictions_inverse[:, 1])
test_rmse_ch4 = np.sqrt(test_mse_ch4)
test_r2_ch4 = r2_score(y_test_inverse[:, 1], test_predictions_inverse[:, 1])

# Save metrics to CSV
metrics = {
    'Metric': ['MSE', 'MAE', 'RMSE', 'R-squared'],
    'Train_CO2': [train_mse_co2, train_mae_co2, train_rmse_co2, train_r2_co2],
    'Train_CH4': [train_mse_ch4, train_mae_ch4, train_rmse_ch4, train_r2_ch4],
    'Test_CO2': [test_mse_co2, test_mae_co2, test_rmse_co2, test_r2_co2],
    'Test_CH4': [test_mse_ch4, test_mae_ch4, test_rmse_ch4, test_r2_ch4]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('Metrics/FNN_metrics.csv', index=False)

# Plotting the results for both CO2 and CH4 for training set
plt.figure(figsize=(9 * cm, 12 * cm), dpi=300)
plt.subplot(2, 1, 1)
plt.plot(y_train_inverse[:, 0], label='Actual CO2 (Train)', color='black', linewidth=0.5, zorder=1)
plt.plot(train_predictions_inverse[:, 0], label='Predicted CO2 (Train)', color='blue', linewidth=0.5, zorder=2)
plt.legend()
plt.title('FNN Predictions vs Actual Data for CO2 (Train)')

plt.subplot(2, 1, 2)
plt.plot(y_train_inverse[:, 1], label='Actual CH4 (Train)', color='black', linewidth=0.5, zorder=1)
plt.plot(train_predictions_inverse[:, 1], label='Predicted CH4 (Train)', color='blue', linewidth=0.5, zorder=2)
plt.legend()
plt.title('FNN Predictions vs Actual Data for CH4 (Train)')
plt.tight_layout()
plt.savefig("Figs/FNN_Training.png", dpi=300)
plt.show()

# Create plots for testing data
plt.figure(figsize=(9 * cm, 12 * cm), dpi=300)
plt.subplot(2, 1, 1)
plt.plot(y_test_inverse[:, 0], label='Actual CO2 (Test)', color='black', linewidth=0.5, zorder=1)
plt.plot(test_predictions_inverse[:, 0], label='Predicted CO2 (Test)', color='blue', linewidth=0.5, zorder=2)
plt.legend()
plt.title('FNN Predictions vs Actual Data for CO2 (Test)')

plt.subplot(2, 1, 2)
plt.plot(y_test_inverse[:, 1], label='Actual CH4 (Test)', color='black', linewidth=0.5, zorder=1)
plt.plot(test_predictions_inverse[:, 1], label='Predicted CH4 (Test)', color='blue', linewidth=0.5, zorder=2)
plt.legend()
plt.title('FNN Predictions vs Actual Data for CH4 (Test)')
plt.tight_layout()
plt.savefig("Figs/FNN_Testing.png", dpi=300)
plt.show()
