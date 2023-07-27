import numpy as np
from pyspark.mllib.linalg import Vectors

def PALearning(X, y, spark, classifier, model_version):
	
	# Fit the pac classifier
	classifier.partial_fit(X, y, classes=np.unique(y))

	predictions = classifier.predict(X)
	
	accuracy = np.count_nonzero(np.array(predictions) == y)/y.shape[0]
	
	print(f"Accuracy of PAC_{model_version}:", accuracy)
	with open(f'./model_accuracies/pac_{model_version}.txt', "a") as ma:
		ma.write(str(accuracy)+'\n')
	
	
	return classifier
