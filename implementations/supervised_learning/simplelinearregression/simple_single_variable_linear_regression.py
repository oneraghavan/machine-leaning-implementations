from math import sqrt

import numpy as np


class SimpleSingleVariableLinearRegression:
    def __init__(self):
        self.intercept = self.slope = 0
        self.step_size = 0.05
        self.tolerance = 0.001
        self.iterations = 0

    def fit(self, input_feature, output):
        self._validate(input_feature, output)

        n = input_feature.size

        converged = False
        iterations = 0
        while not converged:
            iterations += 1
            predictions = input_feature * self.slope + self.intercept

            errors = predictions - output
            gradient_sum_squares = 0
            intercept_derivative = self._feature_derivative(errors, np.ones(n))
            slope_derivative = self._feature_derivative(errors, input_feature)
            gradient_sum_squares = (intercept_derivative * intercept_derivative) + (
                slope_derivative * slope_derivative)
            self.intercept -= self.step_size * intercept_derivative
            self.slope -= self.step_size * slope_derivative

            gradient_magnitude = sqrt(gradient_sum_squares)
            if gradient_magnitude < self.tolerance:
                converged = True

        self.iterations = iterations

    def predit(self, features_predict):
        predictions = [self.slope * i + self.intercept for i in features_predict]
        return predictions

    def _feature_derivative(self, errors, feature):
        derivative = 2 * np.dot(feature, errors)
        return (derivative)

    def _validate(self, input_feature, output):
        if input_feature.size != output.size:
            raise Exception("input feature should be of same lenth as output")
