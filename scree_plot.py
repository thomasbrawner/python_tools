import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def scree_plot(pca_obj, fname=None): 
    '''
    Scree plot for variance & cumulative variance by component from PCA. 

    Arguments: 
        - pca_obj: a fitted sklearn PCA instance
        - fname: path to write plot to file

    Output: 
        - scree plot 
    '''   
    components = pca_obj.n_components_ 
    variance = pca.explained_variance_ratio_
    plt.figure()
    plt.plot(np.arange(1, components + 1), np.cumsum(variance), label='Cumulative Variance')
    plt.plot(np.arange(1, components + 1), variance, label='Variance')
    plt.xlim([0.8, components]); plt.ylim([0.0, 1.01])
    plt.xlabel('No. Components', labelpad=11); plt.ylabel('Variance Explained', labelpad=11)
    plt.legend(loc='best') 
    plt.tight_layout() 
    if fname is not None:
        plt.savefig(fname)
        plt.close() 
    else:
        plt.show() 
    return 


if __name__ == '__main__':
    # load the features from the boston data and do PCA 
    data = load_boston().data
    pca = PCA()
    pca.fit_transform(StandardScaler().fit_transform(data))
    scree_plot(pca, fname='images/boston_scree.png')


