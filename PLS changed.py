# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:44:58 2020

@author: krishnakant
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import stdout
from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split


# Reading the dataset
df = pd.read_excel('Krishnakant.xlsx')
data = np.array(df)
# y = data[:,2:11]#.reshape(-1,1)
y = data[:,2].reshape(-1,1)
x = data[:,11:]
wl = np.arange(450,2451)

# Plotting the spectral data
with plt.style.context('ggplot'):
    plt.plot(wl, x.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Absorbance")
    plt.show()
    
# Removing the samples with different absorbance plot
li = []
for i in range(len(x)):
    if x[i,875]>0.8:
        li.append(i)
x = np.delete(x,li,0)
y = np.delete(y,li,0)

# Plotting the spectral data
with plt.style.context('ggplot'):
    plt.plot(wl, x.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Absorbance")
    
x1 = savgol_filter(x, 11, polyorder=2, deriv=1) # Finding the first order Derivative
x2 = savgol_filter(x, 11, polyorder=2, deriv=2) # Finding the second order Derivative

plt.figure(figsize=(8, 4.5))
with plt.style.context('ggplot'):
    plt.plot(wl, x2.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("D2 Absorbance")
    plt.show()


# Spliting into test and train data
x_train, x_test, y_train, y_test = train_test_split\
    (x2, y, test_size=0.25, random_state=0)


# =============================================================================
# # Normalizing the data
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# 
# y_train = scaler.fit_transform(y_train)
# y_test = scaler.transform(y_test)
# =============================================================================


def optimise_pls_cv(X, y, n_comp, plot_components=True):
    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE '''
    mse = []
    component = np.arange(1, n_comp+1)
    for i in range(1, n_comp+1):
        pls = PLSRegression(n_components=i)
        # Cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=10)
        mse.append(mean_squared_error(y, y_cv))
        comp = 100*(i)/n_comp
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin+1)
    stdout.write("\n")
    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color = 'blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(left=-1)
        plt.show()
    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=msemin+1)
    # Fir to the entire dataset
    pls_opt.fit(X, y)
    y_c = pls_opt.predict(X)
    # Cross-validation
    y_cv = cross_val_predict(pls_opt, X, y, cv=10)
    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
    # Plot regression and figures of merit
    # rangey = max(y) - min(y)
    # rangex = max(y_c) - min(y_c)
    # Fit a line to the CV vs response
    # z = np.polyfit(y, y_c, 1)
    # with plt.style.context(('ggplot')):
    #     fig, ax = plt.subplots(figsize=(9, 5))
    #     ax.scatter(y_c, y, c='red', edgecolors='k')
    #     #Plot the best fit line
    #     ax.plot(np.polyval(z,y), y, c='blue', linewidth=1)
    #     #Plot the ideal 1:1 line
    #     ax.plot(y, y, color='green', linewidth=1)
    #     plt.title('$R^{2}$ (CV): '+str(score_cv))
    #     plt.xlabel('Predicted $^{\circ}$Brix')
    #     plt.ylabel('Measured $^{\circ}$Brix')
    #     plt.show()
    return pls_opt, mse, y_c, y_cv

pls1, mse, y_c, y_cv = optimise_pls_cv(x_train, y_train, 40)


def pls_reg(pls1, x_test, y_test):
    y_c = pls1.predict(x_test)
    # Cross-validation
    # y_cv = cross_val_predict(pls_opt, X, y, cv=10)
    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y_test, y_c)
    # score_cv = r2_score(y_test, y_cv)
    
    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y_test, y_c)
    # mse_cv = mean_squared_error(y_test, y_cv)
    print('\n\nFor Test dataset\nR2 calib: %5.3f'  % score_c)
    # print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    # print('MSE CV: %5.3f' % mse_cv)
    
    return y_c

mmm = pls_reg(pls1, x_test, y_test)





