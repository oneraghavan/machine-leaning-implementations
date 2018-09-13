import numpy as np

# Create a 3 X 20 matrix with random values.
arange = np.arange(0, 40)
samples = np.array([arange * 3, arange * 1])
# Compute the mean vector
mean_x = np.mean(samples[0, :])
mean_y = np.mean(samples[1, :])
# mean_z = np.mean(samples[2,:])
#
mean_vector = np.array([[mean_x], [mean_y]])
# Computation of scatter plot
scatter_matrix = np.zeros((2, 2))
for i in range(samples.shape[1]):
	scatter_matrix += (samples[:, i].reshape(2, 1) - mean_vector).dot((samples[:, i].reshape(2, 1) - mean_vector).T)
print('Scatter Matrix:\n', scatter_matrix)

print('Covariance Matrix:', np.cov(samples))
print('Scatter Matrix:', scatter_matrix)

std_dev_of_x1 = np.std(arange * 3)
std_dev_of_x2 = np.std(arange * -1)

std_dev_products = np.array(
	[[std_dev_of_x1 * std_dev_of_x1, std_dev_of_x1 * std_dev_of_x2],
	 [std_dev_of_x1 * std_dev_of_x2, std_dev_of_x2 * std_dev_of_x2]]
)

print('Covariance Matrix:', np.corrcoef(samples))
print('Std deviation products :', std_dev_products)
print('Covariance Matrix computed from covariance :', np.divide(np.cov(samples), std_dev_products))

# print(samples)
