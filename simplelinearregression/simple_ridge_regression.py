from math import sqrt

import numpy as np


class RidgeRegression:

    def __init__(self):
        self.weights = None
        self.step_size = 1e-22
        self.tolerance = 0.0001
        self.iterations = None
        self.l2_penalty = None
        self.max_iterations = None

    def fit(self, input_feature, output):
        self.validate(input_feature, output)
        input_feature = self.preprocess(input_feature)
        n = input_feature[0].size
        self.weights = np.zeros(n)

        converged = False
        iterations = 0
        while (not converged) and (iterations < self.max_iterations):
        # while not converged:
        #     print "======================="
        #     print "weights" , self.weights
        #     print "======================="
            gradient_sum_squares = 0
            iterations += 1
            predictions = np.dot(input_feature, self.weights)
            errors = predictions - output
            for i in range(len(self.weights)):
                if i ==0:
                    derivative = self.feature_derivative(errors, input_feature[:, 0],self.weights[0],True)
                else:
                    derivative = self.feature_derivative(errors, input_feature[:, i],self.weights[i],False)
                gradient_sum_squares = gradient_sum_squares + (derivative * derivative)
                # print "decent"
                # print self.step_size * derivative
                # print "decent"
                self.weights[i] = self.weights[i] - self.step_size * derivative

            gradient_magnitude = sqrt(gradient_sum_squares)
            # print self.weights
            # print "gradient_magnitude",gradient_magnitude
            # print "======================="
            # print "derivative",derivative
            # print "======================="

            if gradient_magnitude < self.tolerance:
                converged = True
        self.iterations = iterations

    def predit(self, features_predict):
        features_predict = self.preprocess(features_predict)
        predictions = np.dot(features_predict,self.weights)
        return predictions

    def feature_derivative(self, errors, feature,weight,is_constant_feature):
        if(is_constant_feature):
            derivative = 2 * np.dot(feature, errors)
        else:
            derivative = (2 * np.dot(feature, errors)) + (2 * self.l2_penalty * weight)
        return derivative

    def validate(self, input_feature, output):
        if len(input_feature) != output.size:
            raise Exception("input feature should be of same lenth as output")

    def preprocess(self, input_feature):
        return np.insert(input_feature, 0, 1, axis=1)
