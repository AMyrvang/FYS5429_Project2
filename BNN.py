import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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

# Load and preprocess the data
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

class BayesianDenseLayer(tf.keras.layers.Layer):
    """
    Bayesian Dense Layer with KL divergence regularization.
    Attributes:
        units (int): Number of neurons in the layer.
        kl_weight (float): Weight for the KL divergence term.
        activation (callable): Activation function to use.
    """
    def __init__(self, units, kl_weight, activation=None, **kwargs):
        super(BayesianDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kl_weight = kl_weight

    def build(self, input_shape):
        """
        Build the layer, initializing weights.
        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        self.kernel_mu = self.add_weight(name='kernel_mu', shape=(input_shape[-1], self.units),
                                         initializer='glorot_uniform', trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu', shape=(self.units,),
                                       initializer='zeros', trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', shape=(input_shape[-1], self.units),
                                          initializer='zeros', trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho', shape=(self.units,),
                                        initializer='zeros', trainable=True)

    def call(self, inputs, training=None):
        """
        Forward pass through the layer.
        Args:
            inputs (tensor): Input tensor.
            training (bool): Flag indicating training mode.

        Returns:
            tensor: Output tensor after applying the layer transformation.
        """
        kernel_sigma = tf.nn.softplus(self.kernel_rho)
        bias_sigma = tf.nn.softplus(self.bias_rho)
        kernel_dist = tfp.distributions.Normal(loc=self.kernel_mu, scale=kernel_sigma)
        bias_dist = tfp.distributions.Normal(loc=self.bias_mu, scale=bias_sigma)

        if training:
            kernel = kernel_dist.sample()
            bias = bias_dist.sample()
        else:
            kernel = self.kernel_mu
            bias = self.bias_mu

        outputs = self.activation(tf.matmul(inputs, kernel) + bias)
        return outputs

# Model setup
model = tf.keras.Sequential([
    BayesianDenseLayer(100, kl_weight=0.001, activation='relu'),
    tf.keras.layers.Dense(2)  # Two output units for CO2 and CH4
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Model training
model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2)

# Model evaluation
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

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
metrics_df.to_csv('Metrics/BNN_metrics.csv', index=False)

# Print out metrics
print(metrics_df)

# Plotting the results for both CO2 and CH4 for training set
plt.figure(figsize=(9 * cm, 12 * cm), dpi=300)
plt.subplot(2, 1, 1)
plt.plot(y_train_inverse[:, 0], label='Actual CO2 (Train)', color='black', linewidth=0.5, zorder=1)
plt.plot(train_predictions_inverse[:, 0], label='Predicted CO2 (Train)', color='red', linewidth=0.5, zorder=2)
plt.legend()
plt.title('BNN Predictions vs Actual Data for CO2 (Train)')

plt.subplot(2, 1, 2)
plt.plot(y_train_inverse[:, 1], label='Actual CH4 (Train)', color='black', linewidth=0.5, zorder=1)
plt.plot(train_predictions_inverse[:, 1], label='Predicted CH4 (Train)', color='red', linewidth=0.5, zorder=2)
plt.legend()
plt.title('BNN Predictions vs Actual Data for CH4 (Train)')
plt.tight_layout()
plt.savefig("Figs/BNN_Training.png", dpi=300)
plt.show()

# Create plots for testing data
plt.figure(figsize=(9 * cm, 12 * cm), dpi=300)
plt.subplot(2, 1, 1)
plt.plot(y_test_inverse[:, 0], label='Actual CO2 (Test)', color='black', linewidth=0.5, zorder=1)
plt.plot(test_predictions_inverse[:, 0], label='Predicted CO2 (Test)', color='red', linewidth=0.5, zorder=2)
plt.legend()
plt.title('BNN Predictions vs Actual Data for CO2 (Test)')

plt.subplot(2, 1, 2)
plt.plot(y_test_inverse[:, 1], label='Actual CH4 (Test)', color='black', linewidth=0.5, zorder=1)
plt.plot(test_predictions_inverse[:, 1], label='Predicted CH4 (Test)', color='red', linewidth=0.5, zorder=2)
plt.legend()
plt.title('BNN Predictions vs Actual Data for CH4 (Test)')
plt.tight_layout()
plt.savefig("Figs/BNN_Testing.png", dpi=300)
plt.show()

