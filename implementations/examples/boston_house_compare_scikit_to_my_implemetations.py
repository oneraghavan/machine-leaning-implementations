from sklearn import datasets
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from implementations.simplelinearregression import SimpleMultipleFeatureLinearRegression

my_linear_model = SimpleMultipleFeatureLinearRegression()
sci_kit_model = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

X_train = np.array(boston.data[:400])
Y_train = np.array(boston.target[:400])

X_test = np.array(boston.data[401:])
Y_test = np.array(boston.target[401:])

my_linear_model.tolerance = 3000
my_linear_model.step_size = 1e-10
my_linear_model.fit(X_train,Y_train)
sci_kit_model.fit(X_train,Y_train)
predicted_my_model = my_linear_model.predit(X_test)
predicted_sci_kit_model = sci_kit_model.predict(X_test)

print "my model weights" , my_linear_model.weights
print "sci kit model" , sci_kit_model.coef_


my_model_fig, my_model_ax = plt.subplots()
my_model_ax.scatter(Y_test, predicted_my_model)
my_model_ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
my_model_ax.set_xlabel('Measured')
my_model_ax.set_ylabel('Predicted')
plt.show()

sci_kit_fig, sci_kit_ax = plt.subplots()
sci_kit_ax.scatter(Y_test, predicted_sci_kit_model)
sci_kit_ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'b--', lw=4)
sci_kit_ax.set_xlabel('Measured')
sci_kit_ax.set_ylabel('Predicted')
plt.show()