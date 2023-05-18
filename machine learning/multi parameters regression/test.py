import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#loading data
dataframe = pd.read_csv('Advertising.csv')
X = dataframe.values[:,1:4]
X = np.array(X)
Y = np.array(dataframe.values[:,4])

def predict(x,w,b):
    p = np.dot(w,x) +b
    return p

def cost_function(x,y,w,b):
    m = x.shape[0]
    cost =0
    for i in range(m):
        f = np.dot(x[i],w) +b
        cost = cost + (f - y [i])**2
    cost = cost/(2*m)
    return cost

def gardient_function(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

def gardient_descent(X,Y,w_init,b_init,lrate,loop):
    history = []
    w = w_init
    b = b_init
    for i in range(loop):
        dj_db,dj_dw = gardient_function(X,Y,w,b)
        w = w - lrate * dj_dw
        b = b - lrate * dj_db
        history.append(cost_function(X,Y,w,b))
    return w,b,history

w_init = np.zeros(3)
b_init = 5
loop = 1000
lrate = 0.5e-7
w_final,b_final,Cost_history = gardient_descent(X,Y,w_init,b_init,lrate,loop)
plt.scatter(Cost_history,loop,marker ='o')