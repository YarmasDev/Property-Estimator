import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

# Import data
dataset = pd.read_csv("property-prices.csv", delimiter=",")

# Visualize data in a scatter plot
sns.scatterplot(x='sqft_living', y='price', data=dataset)

# Select only numerical data (integers and floats) for correlation
numerical_data = dataset.select_dtypes(include=[np.number])

# Check correlation
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(numerical_data.corr(), annot=True)

# DATA CLEANING
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']

# Load dataset
x = dataset[selected_features]
y = dataset['price']

from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
x_scaled = scaler_x.fit_transform(x)

# Normalize output
y = y.values.reshape(-1, 1)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# Training
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.25)

# Define model 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(7,)))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_hist = model.fit(x_train, y_train, epochs=100, batch_size=50, validation_split=0.2)

epochs_hist.history.keys()

# Plotting 
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model progress')
plt.xlabel('Epochs')
plt.ylabel('Training and validation loss')
plt.legend(['Training loss', 'Validation loss'])

# Make predictions with an example 
x_test_1 = np.array([[4, 3, 1960, 5000, 1, 2000, 3000]])

x_test_scaled_1 = scaler_x.fit_transform(x_test_1)

# Prediction
y_predict_1 = model.predict(x_test_scaled_1)

# The value needs to be re-scaled to be accurate
y_predict_1 = scaler_y.inverse_transform(y_predict_1)
print(y_predict_1)
