import unittest

import numpy as np

from implementations.simplelinearregression import SimpleMultipleFeatureLinearRegression


class TestSimpleSingleFeatureLinearRegression(unittest.TestCase):
    def test_should_raise_exception_when_input_feature_size_not_equal_output_size(self):
        input_feature = np.array([1, 2, 3])
        output = np.array([1, 2, 3, 4])
        simple_model = SimpleMultipleFeatureLinearRegression()
        with self.assertRaises(Exception) as context:
            simple_model.fit(input_feature, output)

        self.assertTrue("input feature should be of same lenth as output" in context.exception)

    def test_should_predict_values_for_the_given_data(self):
        input_feature = np.array([[1, 2, 3], [1, 1, 1], [4, 4, 4]])
        output = np.array([9, 12, 15])
        simple_model = SimpleMultipleFeatureLinearRegression()

        simple_model.fit(input_feature, output)

        actual_predictions = simple_model.predit(np.array([[4, 1, 4]]))
        expected = np.array([14])

        np.testing.assert_almost_equal(actual_predictions, expected, 2)

if __name__ == '__main__':
    unittest.main()

