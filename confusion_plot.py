import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.cross_validation import train_test_split 
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


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

def confusion_plot(observed, predicted, norm=True, model_names=None, fname=None):
    '''
    Plot the results from multiple confusion matrices for the same observed data.

    Arguments:
    ----------
    observed: array of values of dependent variable
    predicted: list of arrays of predicted values from models 
    norm: should the values in the confusion matrix cells be normalized by the sum of within-class observations. Defaults to True. 
    model_names: list of names given to the models to appear in the legend of the plot. 
    fname: path to write file 
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

    if fname is not None: 
        plt.savefig(fname)
        plt.close() 
    else: 
        plt.show() 
    return 


if __name__ == "__main__":

    # generate some fake data, split, and scale 
    X, y = make_classification(n_samples=10000, n_informative=5, n_redundant=6, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    scaler = StandardScaler().fit(X_train)
    X_train_standard = scaler.transform(X_train)
    X_test_standard = scaler.transform(X_test)
    
    # specify classifiers
    ridge = RidgeClassifierCV(alphas=np.logspace(-3, 1))
    lasso = LogisticRegressionCV(Cs=np.logspace(-3, 1))
    forest = RandomForestClassifier(n_estimators=5000, n_jobs=-1) 

    # train the classifiers 
    ridge.fit(X_train_standard, y_train)
    lasso.fit(X_train_standard, y_train)
    forest.fit(X_train, y_train)
   
    # predicted values 
    ridge_preds = ridge.predict(X_test_standard)
    lasso_preds = lasso.predict(X_test_standard)
    forest_preds = forest.predict(X_test)

    # confusion matrices 	
    c1 = confusion_matrix(y_test, ridge_preds)
    c2 = confusion_matrix(y_test, lasso_preds)
    c3 = confusion_matrix(y_test, forest_preds)

    # build a plot to compare results 
    preds = [ridge_preds, lasso_preds, forest_preds]
    names = ['Ridge', 'Lasso', 'Random Forest']
    confusion_plot(y_test, preds, norm=True, model_names=names, fname='images/confusion_plot.png')
