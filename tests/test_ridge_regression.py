import unittest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from simplelinearregression.simple_multiple_feature_linear_regression import SimpleMultipleFeatureLinearRegression
from simplelinearregression.simple_ridge_regression import RidgeRegression


class TestSimpleSingleFeatureLinearRegression(unittest.TestCase):
    def test_should_raise_exception_when_input_feature_size_not_equal_output_size(self):
        input_feature = np.array([1, 2, 3])
        output = np.array([1, 2, 3, 4])
        simple_model = RidgeRegression()
        with self.assertRaises(Exception) as context:
            simple_model.fit(input_feature, output)

        self.assertTrue("input feature should be of same lenth as output" in context.exception)

    def test_should_predict_values_for_the_given_data(self):
        x = np.array([i*np.pi/180 for i in range(60,300,4)])
        final_X = []
        for i in x:
            x_powered_features = [i]
            for pow in range(2,16):
                x_powered_features.append(i**pow)
            final_X.append(x_powered_features)
        final_X = np.array(final_X)
        y = np.sin(x) + np.random.normal(0,0.15,len(x))

        # alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
        alpha_ridge = [1e-15, 1e-10, 1e25]

        # for i in alpha_ridge:
        simple_model= None
        simple_model = RidgeRegression()
        simple_model.l2_penalty = 1e0
        simple_model.step_size = 1.32409e-22
        print simple_model.iterations
        simple_model.fit(final_X, y)

    def create_polynomial_features(self,observation,raise_to_power):
        polynomial_observation = []
        for power in raise_to_power:
            for value in observation:
                polynomial_observation.append(value**power)

        return observation + polynomial_observation

if __name__ == '__main__':
    unittest.main()

