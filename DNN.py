# This code is developed at UW Madison, to provide deep neural network for the department of engineering physics
# This code is prepared by Jun Wang, in Michael L. Corradini's group
# The first application of this code is prediction of CHF for ATF materials

# Part1 import initial required modules
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
import seaborn as sns

# Part2 Read data, and deal with data
# Read the data set
df = pd.read_csv(
    "CHFDATA2.csv",
    na_values=['NA','?'])

# ---- Part2 - Plot the input start
# features = ['Thickness','Angle','Density','ThermalC','SpecificH','Ra',
#             'PV','CHF']
#
# mask = np.zeros_like(df[features].corr(), dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
#
# f, ax = plt.subplots(figsize=(16, 12))
# plt.title('Pearson Correlation Matrix',fontsize=25)
#
# sns.heatmap(df[features].corr(),linewidths=0.25,vmax=1.0,square=True,cmap="BuGn_r",
#             linecolor='w',annot=True,mask=mask,cbar_kws={"shrink": .75})

# ----
# sns.boxplot(y=df["CHF"])
# plt.show()
# ----
# sns.pairplot(df)
# plt.show()
# ----
# print(df.describe())

# ----
# missing=pd.DataFrame({'Missing count':df.isnull().sum()})
# missing.plot.bar()
# ---- Part2 - Plot the input end

# Generate dummies for Substrate material
df = pd.concat([df,pd.get_dummies(df['Oxidized'],prefix="Oxidized")],axis=1)
df.drop('Oxidized', axis=1, inplace=True)

# Generate dummies for Surface condition
df = pd.concat([df,pd.get_dummies(df['Surface'],prefix="Surface")],axis=1)
df.drop('Surface', axis=1, inplace=True)

# Generate dummies for Surface condition
# df = pd.concat([df,pd.get_dummies(df['Materials'],prefix="Materials")],axis=1)
# df.drop('Materials', axis=1, inplace=True)

# # Missing values for Static Contact Angle [deg]
# med = df['Angle'].median()
# df['Angle'] = df['Angle'].fillna(med)
#
# # Missing values for Ra [micron]
# med = df['Ra'].median()
# df['Ra'] = df['Ra'].fillna(med)
#
# # Missing values for PV [micron]
# med = df['PV'].median()
# df['PV'] = df['PV'].fillna(med)
#
# # Missing values for Density
# med = df['Density'].median()
# df['Density'] = df['Density'].fillna(med)
#
# # Missing values for ThermalC
# med = df['ThermalC'].median()
# df['ThermalC'] = df['ThermalC'].fillna(med)
#
# # Missing values for SpecificH
# med = df['SpecificH'].median()
# df['SpecificH'] = df['SpecificH'].fillna(med)

# Standardize ranges
df['Thickness'] = zscore(df['Thickness'])
df['Angle'] = zscore(df['Angle'])
df['Ra'] = zscore(df['Ra'])
df['PV'] = zscore(df['PV'])
df['Density'] = zscore(df['Density'])
df['ThermalC'] = zscore(df['ThermalC'])
df['SpecificH'] = zscore(df['SpecificH'])
# df['Package'] = zscore(df['Package'])

# Convert to numpy - Classification
x_columns = df.columns.drop('TestID').drop('Date').drop('CHF').drop('Materials')
x = df[x_columns].values
y = df['CHF'].values

with open('CHF_x.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(x)

with open('CHF_y.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(y)

# Part3 Build the deep neural network module
# Create train/test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

# import required modules for deep neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

# Build the neural network
model = Sequential()
model.add(Dense(25, input_dim=x.shape[1], activation='relu')) # Hidden 1
model.add(Dense(10, activation='relu')) # Hidden 2
model.add(Dense(1)) # Output
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                        patience=5, verbose=1, mode='auto', restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=1000)

# Part 4 Draw the numerical result
from sklearn import metrics

# Predict
pred = model.predict(x_test)

# Measure MSE error.
score = metrics.mean_squared_error(pred,y_test)
print("Final score (MSE): {}".format(score))

import numpy as np

# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Final score (RMSE): {}".format(score))

# Regression chart.
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

# Plot the chart
chart_regression(pred.flatten(),y_test)

# Part5 Read data, and deal with data
# Read the data set
# df2 = pd.read_csv(
#     "PREDICT.csv",
#     na_values=['NA','?'])
#
# # Standardize ranges
# df2['thickness'] = zscore(df2['thickness'])
# df2['Angle'] = zscore(df2['Angle'])
# df2['Ra'] = zscore(df2['Ra'])
# df2['PV'] = zscore(df2['PV'])
#
# # Convert to numpy - Classification
# x_columns2 = df2.columns.drop('TestID').drop('Date').drop('CHF')
# x2 = df2[x_columns2].values
# y2 = df2['CHF'].values
#
# with open('PREDICT_x.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(x2)
#
# pred2 = model.predict(x2)
# print("Print Predict Data in csv file")
# with open('PREDICT_y.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(pred2)