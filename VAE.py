import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Load the dataset
file_path = 'Data/processed_iskoras_measurements.csv'
data = pd.read_csv(file_path)

# Combine date and time into a single datetime column
data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
data.set_index('datetime', inplace=True)

# Drop the now redundant columns
data.drop(['date', 'time'], axis=1, inplace=True)

# Extract time-related features
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month

# Normalize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Convert back to a DataFrame
data_scaled = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)

# Split the data into training and testing sets
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Prepare the data for the model
X_train = train_data.values.reshape((train_data.shape[0], train_data.shape[1], 1))
X_test = test_data.values.reshape((test_data.shape[0], test_data.shape[1], 1))

def build_encoder(input_shape, latent_dim):
    """
    Build the encoder model.
    Args:
        input_shape (tuple): Shape of the input data.
        latent_dim (int): Dimensionality of the latent space.
    Returns:
        Model: Encoder model.
    """
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(encoder_inputs)
    x = layers.LSTM(32)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
    return encoder

def build_decoder(latent_dim, output_shape):
    """
    Build the decoder model.
    Args:
        latent_dim (int): Dimensionality of the latent space.
        output_shape (tuple): Shape of the output data.
    Returns:
        Model: Decoder model.
    """
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.RepeatVector(output_shape[0])(latent_inputs)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    decoder_outputs = layers.TimeDistributed(layers.Dense(output_shape[1]))(x)
    decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder

class TimeVAE(tf.keras.Model):
    """
    Time-series Variational Autoencoder (VAE) model.
    Args:
        encoder (Model): Encoder model.
        decoder (Model): Decoder model.
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(TimeVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        """
        Call method for the VAE model.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            tensor: Reconstructed input tensor.
        """
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed

    def sampling(self, z_mean, z_log_var):
        """
        Reparameterization trick for sampling.
        Args:
            z_mean (tensor): Mean of the latent space.
            z_log_var (tensor): Log variance of the latent space.

        Returns:
            tensor: Sampled latent vector.
        """
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Hyperparameters
input_shape = (train_data.shape[1], 1)
latent_dim = 16

# Build the encoder and decoder
encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, input_shape)

# Build the VAE
vae = TimeVAE(encoder, decoder)

# Compile the model
vae.compile(optimizer='adam', loss='mse')

# Train the model
vae.fit(X_train, X_train, epochs=1000, batch_size=32, validation_data=(X_test, X_test))

# Make predictions
X_train_pred = vae.predict(X_train)
X_test_pred = vae.predict(X_test)

# Reshape the predictions back to the original shape
X_train_pred = X_train_pred.reshape((X_train_pred.shape[0], X_train_pred.shape[1]))
X_test_pred = X_test_pred.reshape((X_test_pred.shape[0], X_test_pred.shape[1]))

# Inverse transform the scaled data
train_data_inv = scaler.inverse_transform(X_train.reshape((X_train.shape[0], X_train.shape[1])))
test_data_inv = scaler.inverse_transform(X_test.reshape((X_test.shape[0], X_test.shape[1])))
X_train_pred_inv = scaler.inverse_transform(X_train_pred)
X_test_pred_inv = scaler.inverse_transform(X_test_pred)

# Convert back to DataFrame for easier plotting
train_data_inv = pd.DataFrame(train_data_inv, index=train_data.index, columns=train_data.columns)
test_data_inv = pd.DataFrame(test_data_inv, index=test_data.index, columns=test_data.columns)
X_train_pred_inv = pd.DataFrame(X_train_pred_inv, index=train_data.index, columns=train_data.columns)
X_test_pred_inv = pd.DataFrame(X_test_pred_inv, index=test_data.index, columns=test_data.columns)

# Calculate metrics for training set
train_mse_co2 = mean_squared_error(train_data_inv['co2_flux_filtered'], X_train_pred_inv['co2_flux_filtered'])
train_mae_co2 = mean_absolute_error(train_data_inv['co2_flux_filtered'], X_train_pred_inv['co2_flux_filtered'])
train_rmse_co2 = np.sqrt(train_mse_co2)
train_r2_co2 = r2_score(train_data_inv['co2_flux_filtered'], X_train_pred_inv['co2_flux_filtered'])

train_mse_ch4 = mean_squared_error(train_data_inv['ch4_flux_filtered'], X_train_pred_inv['ch4_flux_filtered'])
train_mae_ch4 = mean_absolute_error(train_data_inv['ch4_flux_filtered'], X_train_pred_inv['ch4_flux_filtered'])
train_rmse_ch4 = np.sqrt(train_mse_ch4)
train_r2_ch4 = r2_score(train_data_inv['ch4_flux_filtered'], X_train_pred_inv['ch4_flux_filtered'])

# Calculate metrics for test set
test_mse_co2 = mean_squared_error(test_data_inv['co2_flux_filtered'], X_test_pred_inv['co2_flux_filtered'])
test_mae_co2 = mean_absolute_error(test_data_inv['co2_flux_filtered'], X_test_pred_inv['co2_flux_filtered'])
test_rmse_co2 = np.sqrt(test_mse_co2)
test_r2_co2 = r2_score(test_data_inv['co2_flux_filtered'], X_test_pred_inv['co2_flux_filtered'])

test_mse_ch4 = mean_squared_error(test_data_inv['ch4_flux_filtered'], X_test_pred_inv['ch4_flux_filtered'])
test_mae_ch4 = mean_absolute_error(test_data_inv['ch4_flux_filtered'], X_test_pred_inv['ch4_flux_filtered'])
test_rmse_ch4 = np.sqrt(test_mse_ch4)
test_r2_ch4 = r2_score(test_data_inv['ch4_flux_filtered'], X_test_pred_inv['ch4_flux_filtered'])

metrics = {
    'Metric': ['MSE', 'MAE', 'RMSE', 'R-squared'],
    'Train_CO2': [train_mse_co2, train_mae_co2, train_rmse_co2, train_r2_co2],
    'Train_CH4': [train_mse_ch4, train_mae_ch4, train_rmse_ch4, train_r2_ch4],
    'Test_CO2': [test_mse_co2, test_mae_co2, test_rmse_co2, test_r2_co2],
    'Test_CH4': [test_mse_ch4, test_mae_ch4, test_rmse_ch4, test_r2_ch4]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('Metrics/VAE_metrics.csv', index=False)

# Reset index for plotting using row numbers
train_data_inv_reset = train_data_inv.reset_index(drop=True)
X_train_pred_inv_reset = X_train_pred_inv.reset_index(drop=True)
test_data_inv_reset = test_data_inv.reset_index(drop=True)
X_test_pred_inv_reset = X_test_pred_inv.reset_index(drop=True)

# Plotting the results for both CO2 and CH4 for training set
plt.figure(figsize=(9 * cm, 12 * cm), dpi=300)
plt.subplot(2, 1, 1)
plt.plot(train_data_inv_reset['co2_flux_filtered'], label='Actual CO2 (Train)', color='black', linewidth=0.5, zorder=1)
plt.plot(X_train_pred_inv_reset['co2_flux_filtered'], label='Predicted CO2 (Train)', color='orange', linewidth=0.5, zorder=2)
plt.legend()
plt.title('VAE Predictions vs Actual Data for CO2 (Train)')

plt.subplot(2, 1, 2)
plt.plot(train_data_inv_reset['ch4_flux_filtered'], label='Actual CH4 (Train)', color='black', linewidth=0.5, zorder=1)
plt.plot(X_train_pred_inv_reset['ch4_flux_filtered'], label='Predicted CH4 (Train)', color='orange', linewidth=0.5, zorder=2)
plt.legend()
plt.title('VAE Predictions vs Actual Data for CH4 (Train)')
plt.tight_layout()
plt.savefig("Figs/VAE_Training.png", dpi=300)


# Plotting the results for both CO2 and CH4 for test set
plt.figure(figsize=(9 * cm, 12 * cm), dpi=300)
plt.subplot(2, 1, 1)
plt.plot(test_data_inv_reset['co2_flux_filtered'], label='Actual CO2 (Test)', color='black', linewidth=0.5, zorder=1)
plt.plot(X_test_pred_inv_reset['co2_flux_filtered'], label='Predicted CO2 (Test)', color='orange', linewidth=0.5, zorder=2)
plt.legend()
plt.title('VAE Predictions vs Actual Data for CO2 (Test)')

plt.subplot(2, 1, 2)
plt.plot(test_data_inv_reset['ch4_flux_filtered'], label='Actual CH4 (Test)', color='black', linewidth=0.5, zorder=1)
plt.plot(X_test_pred_inv_reset['ch4_flux_filtered'], label='Predicted CH4 (Test)', color='orange', linewidth=0.5, zorder=2)
plt.legend()
plt.title('VAE Predictions vs Actual Data for CH4 (Test)')
plt.tight_layout()
plt.savefig("Figs/VAE_Testing.png", dpi=300)
