
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.metrics import confusion_matrix


def format_confusion_matrix(cmat):
	'''
	Transfrom a confusion matrix from sklearn.metrics to a long-form 
	data frame with observed and predicted labels. 
	'''
	cmat = pd.DataFrame(cmat, index=['False', 'True'], columns=['False', 'True'])
	cmat.index.name = 'Observed'
	cmat.columns.name = 'Predicted'
	cmat.reset_index(inplace=True)
	cmat = pd.melt(cmat, id_vars='Observed')
	return cmat 

def confusion_plot(observed, predicted, norm=True, model_names=None):
	'''
	Plot the results from multiple confusion matrices for the same observed data.

	Arguments:
	----------
		observed: array of values of dependent variable
		predicted: list of arrays of predicted values from models 
		norm: should the values in the confusion matrix cells be normalized by 
			the sum of within-class observations. Defaults to True. 
		model_names: list of names given to the models to appear in the legend 
			of the plot. 
	'''
	cmats = [confusion_matrix(observed, pred) for pred in predicted]
	if norm:
		cmats = [cmat / cmat.sum(axis=1).astype(float) for cmat in cmats]
	cmats = [format_confusion_matrix(cmat) for cmat in cmats]

	if model_names is not None:
		if len(model_names) != len(predicted):
			raise Exception('List of model names should be of same length as the list of predicted values.')
		for name, cmat in zip(model_names, cmats):
			cmat[' '] = name 
	else:
		for idx, cmat in zip(np.arange(len(cmats)), cmats):
			cmat[' '] = idx
	
	cdata = pd.concat(cmats)

	f, (ax0, ax1) = plt.subplots(2, sharex=True)
	sns.barplot(x='Predicted', y='value', hue=' ', data=cdata.query('Observed == "True"'), ax=ax0)
	ax0.set_ylabel('True', labelpad=12); ax0.set_xlabel('')
	if norm:
		ax0.set_ylim([0.0, 1.0])
	else:
		ax0.set_ylim([0.0, cdata.query('Observed == "True"')['value'].max() + 1])
	sns.barplot(x='Predicted', y='value', hue=' ', data=cdata.query('Observed == "False"'), ax=ax1)
	ax1.set_ylabel('False', labelpad=12); ax1.set_xlabel('Predicted', labelpad=12)
	if norm:
		ax1.set_ylim([0.0, 1.0])
	else:
		ax1.set_ylim([0.0, cdata.query('Observed == "False"')['value'].max() + 1])
	f.subplots_adjust(hspace=0.02)
	plt.tight_layout() 


if __name__ == "__main__":

	# fake results 
	data = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1])
	model1 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
	model2 = np.array([1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0])
	model3 = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0])

	# confusion matrices
	c1 = confusion_matrix(data, model1)
	c2 = confusion_matrix(data, model2)
	c3 = confusion_matrix(data, model3)

	# build a plot to compare results 
	preds = [model1, model2, model3]
	names = ['Model 1', 'Model 2', 'Model 3']
	confusion_plot(data, preds, norm=True, model_names=names)
