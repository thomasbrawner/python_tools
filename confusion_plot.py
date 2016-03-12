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
sns.set_style('white')


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

def confusion_plot(observed, predicted, fname, model_names=None):
    '''
    Plot the results from multiple confusion matrices for the same observed data. Counts in each 
    cell are normalized by the observed within-class totals. 

    Arguments:
    ----------
    observed: array of values of dependent variable
    predicted: list of arrays of predicted values from models 
    model_names: list of names given to the models to appear in the legend of the plot. 
    fname: path to write file 
    '''
    cmats = [confusion_matrix(observed, pred) for pred in predicted]
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
    index = np.arange(len(predicted))
    width = 0.8
    cs = np.linspace(0.35, 0.65, num=len(predicted))
    cs = list(reversed([str(x) for x in cs]))
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, subplot_kw={'ylim' : [0, 1], 'xlim' : [-0.6, len(predicted) - 0.4]})
    
    tp = cdata.query('Observed == "True" & Predicted == "True"')['value']
    ax1.bar(index, tp.values, align='center', width=width, color=cs)
    ax1.set_xticks([])
    ax1.set_title('True Positives', y=1.01)
    ax1.set_xlabel(''); ax1.set_ylabel('')
    
    fn = cdata.query('Observed == "True" & Predicted == "False"')['value']
    ax2.bar(index, fn.values, align='center', width=width, color=cs)
    ax2.set_xticks([])
    ax2.set_title('False Negatives', y=1.01)
    ax2.set_xlabel(''); ax2.set_ylabel('')
    
    fp = cdata.query('Observed == "False" & Predicted == "True"')['value']
    ax3.bar(index, fp.values, align='center', width=width, color=cs)
    ax3.set_xticks([])
    ax3.set_title('False Positives', y=1.01)
    ax3.set_xlabel(''); ax3.set_ylabel('')
    
    tn = cdata.query('Observed == "False" & Predicted == "False"')['value']
    b1, b2, b3 = ax4.bar(index, tn.values, align='center', width=width, color=cs)
    ax4.set_xticks([])
    ax4.set_title('True Negatives', y=1.01)
    ax4.set_xlabel(''); ax4.set_ylabel('')
   
    lgd = ax4.legend((b1, b2, b3), model_names, ncol=len(model_names), fontsize=12, bbox_to_anchor=(0.5, -0.05), fancybox=True)
    plt.tight_layout() 
    plt.savefig(fname, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close() 


if __name__ == "__main__":

    # generate some fake data, split, and scale 
    X, y = make_classification(n_samples=1000, n_informative=5, n_redundant=6, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    scaler = StandardScaler().fit(X_train)
    X_train_standard = scaler.transform(X_train)
    X_test_standard = scaler.transform(X_test)
    
    # specify classifiers
    ridge = RidgeClassifierCV(alphas=np.logspace(-3, 1, 20))
    lasso = LogisticRegressionCV(Cs=np.logspace(-3, 1, num=20))
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1) 

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
    confusion_plot(y_test, preds, model_names=names, fname='images/confusion_plot.png') 
