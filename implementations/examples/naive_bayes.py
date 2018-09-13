from __future__ import division, print_function

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from implementations.supervised_learning.naive_bayes import NaiveBayes


def main():
	data = datasets.load_digits()
	X = normalize(data.data)
	y = data.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

	# print(X_test[0])
	clf = NaiveBayes()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	# print(y_test, y_pred)

	accuracy = accuracy_score(y_test, y_pred)

	print("Accuracy:", accuracy)

if __name__ == "__main__":
	main()
