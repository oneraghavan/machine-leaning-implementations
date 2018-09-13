import math
import numpy as np

class NaiveBayes():

	def __init__(self):
		self.classes = []
		self.MEAN_VAR = {}
		self.P_OF_Y = {}

	def fit(self, X, Y):
		self.X, self.Y = X, Y
		self.classes = np.unique(self.Y.tolist())
		for cls in set(self.classes):
			self.P_OF_Y[cls] = float(  np.count_nonzero(self.classes == cls) / float(len(self.classes)))
			self.MEAN_VAR[cls] = []
			x_with_class_c = self.X[np.where(Y == cls)]
			for var in x_with_class_c.T:
				self.MEAN_VAR[cls].append({"mean": var.mean(), "variance": var.var()})

	def _log_likelihood(self, mean, var, x):
		""" Gaussian likelihood of the data x given mean and var """
		eps = 1e-4  # Added in denominator to prevent division by zero
		coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
		exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
		return coeff * exponent

	def _classify(self, X):
		# P(Y|X) = P(X|Y) P(Y) / P(X)
		# P(X|Y)
		#       Likelihood of the feature being in class Y
		# P(Y)
		#     Prior of the class
		# P(X)
		#     Prior of the predictor

		posteriors = []
		for class_ in self.classes:
			posterior = self.P_OF_Y[class_]
			col_pos = 0
			for col in X:
				posterior *= self._log_likelihood(self.MEAN_VAR[class_][col_pos]["mean"],self.MEAN_VAR[class_][col_pos]["variance"],col)
				col_pos += 1
			posteriors.append((class_, posterior))

		return sorted(posteriors, key=lambda x: -1 * x[1])[0][0]

	def predict(self, X):
		return [self._classify(x) for x in X]
