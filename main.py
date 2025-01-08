import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error, r2_score
diabetes = datasets.load_diabetes()
# print(diabetes.data)
# print (diabetes.target)
diabetes_X= diabetes.data[:,np.newaxis,2]
# print (diabetes_X)
diabetes_X_train=diabetes_X[:-30]
diabetes_X_test = diabetes_X[-20:]
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-20:]
model = linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predicted  = model.predict(diabetes_X_test)

c =  print ('mean square error is :', mean_squared_error(diabetes_Y_test , diabetes_Y_predicted))
print ('weight :', model.coef_)
print ('intercept :', model.intercept_)
plt.scatter(diabetes_X_test,diabetes_Y_test,color='black')
plt.plot(diabetes_X_test,diabetes_Y_predicted,color='blue',linewidth=3)
plt.show()
# diabetes_X_train=diabetes_X[:-20]
# diabetes_X_test=diabetes_X[-20:]
# diabetes_y_train=diabetes.target[:-20]
# diabetes_y_test=diabetes.target[-20:]
# model = linear_model.LinearRegression()
# model.fit(diabetes_X_train,diabetes_y_train)
# print(model.predict(diabetes_X_test))
# print(diabetes_y_test)
# plt.scatter(diabetes_X_test,diabetes_y_test,color='black')
#

