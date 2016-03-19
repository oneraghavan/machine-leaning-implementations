import unittest
import numpy as np
from simplelinearregression.simple_single_variable_linear_regression import SimpleSingleVariableLinearRegression



class TestSimpleSingleFeatureLinearRegression(unittest.TestCase):
    def test_should_raise_exception_when_input_feature_size_not_equal_output_size(self):
        input_feature = np.array([1, 2, 3])
        output = np.array([1, 2, 3, 4])
        simple_model = SimpleSingleVariableLinearRegression()
        with self.assertRaises(Exception) as context:
            simple_model.fit(input_feature, output)

        self.assertTrue("input feature should be of same lenth as output" in context.exception)

    def test_should_predict_values_for_the_given_data(self):
        input_feature = np.array([1, 2, 3])
        output = np.array([3, 5, 7])
        simple_model = SimpleSingleVariableLinearRegression()

        simple_model.fit(input_feature, output)

        actual_predictions = simple_model.predit(np.array([4, 5, 6]))
        expected = np.array([9, 11, 13])

        np.testing.assert_almost_equal(actual_predictions, expected, 2)

if __name__ == '__main__':
    unittest.main()
