import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import  linear_model 
import pandas as pd 

names=[0,1,2,3,4]
data = pd.read_csv("iris.csv",names=names)
data=data.iloc[:,:4]


#================================================= Linear Regression ============================================
#print(data.head())
y= data[0] 
x = data[1] 
#print(x)
   
x=x.reshape(len(x),1) 
y=y.reshape(len(y),1) 


# x_train = x[:100] 
# x_test = x[100:]

# y_train = y[:100] 
# y_test = y[100:]


train=[]
x_test = []
y_test = []

for i in range(0,150,10):
	train.append(i)
	x_test.append(x[i])
	y_test.append(y[i])

for i in range(0,150,7):
	train.append(i)
	x_test.append(x[i])
	y_test.append(y[i])
    
x_train = x
x_train=np.delete(x,train)
x_train=x_train.reshape(len(x_train),1)

x_test=np.array(x_test).reshape(len(x_test),1)
   
y_train = y
y_train=np.delete(y,train)
y_train=y_train.reshape(len(y_train),1)

y_test=np.array(y_test).reshape(len(y_test),1)
   
  

regr = linear_model.LinearRegression() 
   
 
regr.fit(y_train, x_train) 
   

predict=regr.predict(y_test)
#print(predict)

error=0

for i in range(len(predict)):
	error+=(x_test[i]-predict[i])**2

print("Single Variate RMS error - ",(error/len(train))**.5)
print()

#========================= Multiple Regression ===============================
 


x_test=[]
x_train=[]

y_test=[]
y_train=[]

#print(y_test)

for i in range(150):
	if (i%7)==0:
		y_test.append([data.iloc[i][0], data.iloc[i][1]])
		x_test.append([data.iloc[i][2]])

	elif (i%10)==0:
		y_test.append([data.iloc[i][0], data.iloc[i][1]])
		x_test.append([data.iloc[i][2]])

	else:
		y_train.append([data.iloc[i][0], data.iloc[i][1]])
		x_train.append([data.iloc[i][2]])


x_test=np.array(x_test).reshape(len(x_test),1)
  

regr = linear_model.LinearRegression() 
   
 
regr.fit(y_train, x_train) 
   

predict=regr.predict(y_test)
#print(predict)

error=0

for i in range(len(predict)):
	error+=(x_test[i]-predict[i])**2

print("Multi Variate RMS error - ",(error/len(train))**.5)
#print()

 
