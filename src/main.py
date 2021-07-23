import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
import pickle

#################################################################################

df = pd.read_excel("Revised Data_edited.xlsx", sheet_name = "Sheet1")

df.drop(["Borehole","C", "SPT N"],axis=1,inplace= True)
df = df.drop([95,101])

#################################################################################

def SVM_rbf(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
    
    svr = SVR(kernel = "rbf", C = 10)
    
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    scaler = MinMaxScaler()
    scaler.fit(X_tr)

    X_tr_scaled = scaler.transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    svr.fit(X_tr_scaled, y_tr)
    ypred_svr = svr.predict(X_test_scaled)
    ypred_svr_train = svr.predict(X_tr_scaled)
    print("r2-score for training is : {:0.4f}".format(r2_score(y_tr,ypred_svr_train)))
    print("RMSE for training is : {:0.4f}".format(MSE(y_tr,ypred_svr_train)**0.5))
    print("MAE for training is : {:0.4f}".format(MAE(y_tr,ypred_svr_train)))
    
    print("\nr2-score for testing is : {:0.4f}".format(r2_score(y_test,ypred_svr)))
    print("RMSE for testing is : {:0.4f}".format(MSE(y_test,ypred_svr)**0.5))
    print("MAE for testing is : {:0.4f}".format(MAE(y_test,ypred_svr)))
    
    plt.scatter(y_test,ypred_svr)
    plt.show()
    
    filename = 'finalized_model_SVM.sav'
    pickle.dump(svr, open(filename, 'wb'))
    filename2 = 'scaler_SVM.sav'
    pickle.dump(scaler, open(filename2, 'wb'))
    
#################################################################################

X = df[["N60","Depth","D30 ","D60", "Silt "]]
y = df["Phi"]

SVM_rbf(X,y)

