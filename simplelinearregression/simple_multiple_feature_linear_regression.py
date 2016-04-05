from math import sqrt

import numpy as np

class SimpleMultipleFeatureLinearRegression:
    def __init__(self):
        self.weights = None
        self.step_size = 1e-10
        self.tolerance = 0.001
        self.iterations = None

    def fit(self, input_feature, output):
        self.validate(input_feature, output)
        input_feature = self.preprocess(input_feature)
        n = input_feature[0].size
        self.weights = np.zeros(n)

        converged = False
        iterations = 0
        while not converged:
            gradient_sum_squares = 0
            iterations += 1
            predictions = np.dot(input_feature, self.weights)
            errors = predictions - output
            for i in range(len(self.weights)):
                derivative = self.feature_derivative(errors, input_feature[:, i])
                gradient_sum_squares = gradient_sum_squares + (derivative * derivative)
                self.weights[i] = self.weights[i] - self.step_size * derivative

            gradient_magnitude = sqrt(gradient_sum_squares)
            print self.weights
            print gradient_magnitude
            if gradient_magnitude < self.tolerance:
                converged = True
        self.iterations = iterations

    def predit(self, features_predict):
        features_predict = self.preprocess(features_predict)
        print self.weights
        predictions = np.dot(features_predict,self.weights)
        return predictions

    def feature_derivative(self, errors, feature):
        derivative = 2 * np.dot(feature, errors)
        return (derivative)

    def validate(self, input_feature, output):
        if len(input_feature) != len(output):
            raise Exception("input feature should be of same lenth as output")

    def preprocess(self, input_feature):
        return np.insert(input_feature, 0, 1, axis=1)
