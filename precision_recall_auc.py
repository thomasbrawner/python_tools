# precision_recall_auc.py
# tb 17 nov 2015

import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from sklearn.metrics import auc, precision_recall_curve


def recall_precision_auc(observed, predicted, plot=False, fname=None): 
	'''
	Evaluate the precision-recall curve for a classifier and return tuple of 
	precision, recall, and auc. Optional plot. 

	Arguments:
	----------
		- observed: array of observed classes
		- predicted: array of predicted classes 
		- plot: boolean, plot the resulting curve 
		- fname: path for writing plot to file, writes to file even where plot=False

	Values:
	-------
		- (precision, recall, auc)
	'''
	precision, recall, thresholds = precision_recall_curve(observed, predicted)
	pr_auc = auc(recall, precision)

	if (plot or fname is not None):
		plt.figure() 
		plt.plot(recall, precision, linestyle='-', color='k')
		plt.xlabel('Recall', labelpad=11); plt.ylabel('Precision', labelpad=11)
		plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
		plt.tight_layout() 

		if fname is not None: 
			plt.savefig(fname)
			plt.close() 
		else: 
			plt.show() 

	return recall, precision, pr_auc


if __name__ == "__main__":

	# fake results 
	data = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
	pred = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
	recall, precision, pr_auc = recall_precision_auc(data, pred)
