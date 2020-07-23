# creator Krishnakant

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
y = data[:,5].reshape(-1,1)
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
    
x1 = savgol_filter(x, 21, polyorder=2, deriv=1) # Finding the first order Derivative
x2 = savgol_filter(x, 21, polyorder=2, deriv=2) # Finding the second order Derivative

plt.figure(figsize=(8, 4.5))
with plt.style.context('ggplot'):
    plt.plot(wl, x1.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("D2 Absorbance")
    plt.show()


# Spliting into test and train data
x_train, x_test, y_train, y_test = train_test_split\
    (x1, y, test_size=0.25, random_state=42)


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


def pls_variable_selection(X, y, max_comp):
    
    # Define MSE array to be populated
    mse = np.zeros((max_comp,X.shape[1]))
    # Loop over the number of PLS components
    for i in range(max_comp):
        
        # Regression with specified number of components, using full spectrum
        pls1 = PLSRegression(n_components=i+1, max_iter = 99999999)
        pls1.fit(X, y)
        
        # Indices of sort spectra according to ascending absolute value of PLS coefficients
        sorted_ind = np.argsort(np.abs(pls1.coef_[:,0]))
        # Sort spectra accordingly 
        Xc = X[:,sorted_ind]
        # Discard one wavelength at a time of the sorted spectra,
        # regress, and calculate the MSE cross-validation
        for j in range(Xc.shape[1]-(i+1)):
            pls2 = PLSRegression(n_components=i+1, max_iter = 99999999)
            # pls2.fit(Xc[:, j:], y)
            
            y_cv = cross_val_predict(pls2, Xc[:, j:], y, cv=5, n_jobs =4, verbose=0)
            mse[i,j] = mean_squared_error(y, y_cv)
    
        comp = 100*(i+1)/(max_comp)
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
    # # Calculate and print the position of minimum in MSE
    mseminx,mseminy = np.where(mse==np.min(mse[np.nonzero(mse)]))
    print("Optimised number of PLS components: ", mseminx[0]+1)
    print("Wavelengths to be discarded ",mseminy[0])
    print('Optimised MSEP ', mse[mseminx,mseminy][0])
    stdout.write("\n")
    # plt.imshow(mse, interpolation=None)
    # plt.show()
    # Calculate PLS with optimal components and export values
    pls = PLSRegression(n_components=mseminx[0]+1)
    pls.fit(X, y)
        
    sorted_ind = np.argsort(np.abs(pls.coef_[:,0]))
    Xc = X[:,sorted_ind]
    return(mse,Xc[:,mseminy[0]:],mseminx[0]+1,mseminy[0], sorted_ind)

mse, x_c, n_comp, n_param, sorted_ind = \
    pls_variable_selection(x_train, y_train, 20)

def plot_MSE(mse):
    no_of_plots = mse.shape[0]
    k = int(np.power(no_of_plots,0.5))+1
    print(k)
    fig = plt.figure(figsize = (15,18))
    for i in range(no_of_plots):
        plt.subplot(k,k,i+1)
        plt.plot(range(mse.shape[1]), mse[i], label = f'{i+1}th component')
        plt.title(f'for n_components = {i+1}')
        plt.ylabel('MSE')
        plt.xlabel('No of wavelengths removed')

plot_MSE(mse)

def pls_reg(X, y, n_comp, x_test, y_test, sorted_ind, n_param):
    pls_opt = PLSRegression(n_components= n_comp)

    pls_opt.fit(X, y)
    y_c = pls_opt.predict(X)

    # Cross-validation
    y_cv = cross_val_predict(pls_opt, X, y, cv=10, n_jobs=4, verbose=0)
    
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
    
    pls_opt.fit(X, y)
    x_test_changed = x_test[:,sorted_ind]
    x_test_changed = x_test_changed[:,n_param:]
    y_c = pls_opt.predict(x_test_changed)
    
    # Calculate MSE and r2_score for test data
    score_c = r2_score(y_test, y_c)
    mse_c = mean_squared_error(y_test, y_c)
    
    print('\n\nR2 calib: %5.3f'  % score_c)
    print('MSE calib: %5.3f' % mse_c)
    
    return y_c

mmm = pls_reg(x_c, y_train, n_comp, x_test, y_test, sorted_ind, n_param)


ix = np.in1d(wl.ravel(), wl[sorted_ind][:n_param])
import matplotlib.collections as collections
# Plot spectra with superimpose selected bands
fig, ax = plt.subplots(figsize=(8,9))
with plt.style.context(('ggplot')):
    ax.plot(wl, x1.T)
    plt.ylabel('First derivative absorbance spectra')
    plt.xlabel('Wavelength (nm)')
collection = collections.BrokenBarHCollection.span_where(
    wl, ymin=-1, ymax=1, where=ix == True, facecolor='red', alpha=0.3)
ax.add_collection(collection)
plt.show()


