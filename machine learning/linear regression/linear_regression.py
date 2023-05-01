import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('Advertising.csv')
X = dataframe.values[:,2]
Y = dataframe.values[:,4]
print(Y)
plt.scatter(X,Y,marker ='o')

def predict(x,weight,bias) :
    return weight*x+bias

def cost_function(X,Y,weight,bias):
    n = len(X)
    sum_cost = 0.0
    for i in range(n):
        sum_cost += (Y[i]-(X[i]*weight + bias))**2
    return sum_cost/n    
def update(X,Y,weight,bias,learning_rate):
    n = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2*X[i]*(Y[i]-(X[i]*weight+bias))
        bias_temp += -2*(Y[i]-(X[i]*weight+bias))
    weight -= (weight_temp/n)*learning_rate
    bias -= (bias_temp/n)*learning_rate
    return weight,bias

def learning(X,Y,weight,bias,learning_rate,iter):
    cos_his = []
    for i in range(iter):
        weight,bias = update(X,Y,weight,bias,learning_rate)
        cost = cost_function(X,Y,weight,bias)
        cos_his.append(cost)
    
    return weight,bias,cos_his
weight,bias,cost = learning(X,Y,0.03,0.0014,0.001,3)
loop = [i for i in range(30)]
#plt.plot(cost,loop)
#plt.show()
plt.plot(X,predict(X,weight,bias))
new = float(input("enter:"))
print("predict:",predict(new,weight,bias))
        