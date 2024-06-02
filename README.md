# FYS5429 Project 2
This project aims to forecast greenhouse gas emissions from thawing permafrost using neural network models. We explore three different models: FNN, BNN, and VAE, using data from the Iskoras permafrost peatlands in Northern Norway.  

### Requirements
To run the Python programs, the following Python packages must be installed:
- Numpy
- Pandas
- Tensorflow
- Tensorflow_probability
- Keras
- Scikit-learn
- Matplotlib
- Seaborn

### Structure
- `process_data.py`: Script for preprocessing the dataset, including renaming variables, converting CH4 flux measurements, removing rows with missing values, and normalizing the data.
- `FNN.py`: Contains the implementation of the Feedforward Neural Network, using TensorFlow and Keras for predicting CO2 and CH4 emissions.
- `BNN.py`: Contains the implementation of the Bayesian Neural Network, using TensorFlow and TensorFlow Probability to provide probabilistic predictions.
- `VAE.py`: Contains the implementation of the Variational Autoencoder, using TensorFlow and Keras to model the complex interdependencies within the emissions data.

### Run code
To successfully execute the code, please note that you might need to modify the file path in the script to correctly access the data file located in the 'Data' folder. Ensure that all required packages are installed, and then enter the following command in the terminal to run the codes: 

```bash
python3 process_data.py
```
```bash
python3 FNN.py
```
```bash
python3 BNN.py
```
```bash
python3 VAE.py
```

### Dataset
The dataset used in this project can be found here: [Dataset](https://zenodo.org/records/7913027).