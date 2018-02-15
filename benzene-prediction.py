from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras import regularizers
from keras import backend as K
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# https://archive.ics.uci.edu/ml/datasets/Air+Quality

"""
data set columns :
0 Date (DD/MM/YYYY)
1 Time (HH.MM.SS)
2 True hourly averaged concentration CO in mg/m^3 (reference analyzer)
3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
7 True hourly averaged NOx concentration in ppb (reference analyzer)
8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
12 Temperature in °C
13 Relative Humidity (%)
14 AH Absolute Humidity

 Missing values are tagged with -200 value. 
"""

# what and where
data_dir = "aqm_data"
airquality_file = os.path.join(data_dir, "AirQualityUCI.csv")

# Data Clean-up

# read in that strange European CSV data, semi-colon separated, with commas for decimal points
aqdf = pd.read_csv(airquality_file, sep=";", decimal=",", header=0)

# remove date, time, and last two NaN columns using drop
# plus, looking T the Non Metanic HydroCarbons column, NMHC(GT), it barely has anything useful so drop it
aqdf.drop(["Date", "Time", "NMHC(GT)", "Unnamed: 15", "Unnamed: 16"], axis=1, inplace=True)

# at lot of all NaN rows at the end - drop these
aqdf.dropna(how='all', inplace=True)

# replace any -200 values with NaN
aqdf.replace(-200, np.nan, inplace=True)

# fill NaNs in each column with the mean value of the column
aqdf.fillna(aqdf.mean(), inplace=True)

# Normalize

# convert the frame to its Numpy matrix representation
x_orig = aqdf.as_matrix()

# scale each column: subtract from each column the mean of the column and divide by its standard deviation
# z = (x - mean)/ std; but use StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_orig)

# save for later for predictions with unseen data
x_means = scaler.mean_
x_stds = scaler.scale_

# Prepare training and test data

# Note Benzine (C6H6) is now column 2
y = x_scaled[:, 2]                  # y is just Benzine column - the target
x = np.delete(x_scaled, 2, axis=1)  # x is everything else - the input

# split data into first 70% for training, and last 30% for testing
# a bit crude - should use cross_validation.train_test_split
train_size = int(0.7 * x.shape[0])
x_train, x_test, y_train, y_test = x[0:train_size], x[train_size:], y[0:train_size], y[train_size:]

# Set up and run the model

# autoencoder like structure 11 features -> 8 latent space -> 1 output
readings = Input(shape=(11, ))
encoded = Dense(8, activation='relu', kernel_initializer='glorot_uniform')(readings)
decoded = Dense(1, kernel_initializer='glorot_uniform')(encoded)

model = Model(inputs=[readings], outputs=[decoded])
model.compile(loss='mse', optimizer='adam')

my_epochs = 50
my_batch_size = 10

history = model.fit(x_train, y_train, batch_size=my_batch_size, epochs=my_epochs, validation_split=0.2)
y_test_pred = model.predict(x_test).flatten()

# Look at the results

restore_benzine = lambda m: (m * x_stds[2]) + x_means[2]

for i in range(10):
    label = restore_benzine(y_test[i])
    prediction = restore_benzine(y_test_pred[i])
    print("Benzene Conc. expected: {:.3f}, predicted: {:.3f}".format(label, prediction))

plt.figure(figsize=(12, 6), dpi=100)
plt.title("Entire C6H6 test set values against predictions")
plt.plot(np.arange(y_test.shape[0]), restore_benzine(y_test), color='b', label='actual')
plt.plot(np.arange(y_test_pred.shape[0]), restore_benzine(y_test_pred), color='r', alpha=0.5, label='predicted')
plt.xlabel("time")
plt.ylabel("C6H6 concentrations (microg/m^3)")
plt.legend(loc='best')
plt.show()

K.clear_session()
