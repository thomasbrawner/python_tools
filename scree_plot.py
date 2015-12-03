import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def scree_plot(pca_obj, fname=None): 
    '''
    Scree plot for principal components analysis objects. 

    Arguments: 
        - pca_obj: a fitted sklearn PCA instance
        - fname: path to write plot to file

    Output: 
        - scree plot 
    '''    
    plt.figure()
    plt.plot(np.arange(pca_obj.n_components_ + 1), np.hstack([0, np.cumsum(pca_obj.explained_variance_ratio_)]))
    plt.xlim([0, pca.n_components_]); plt.ylim([0.0, 1.01])
    plt.xlabel('No. Components', labelpad=11); plt.ylabel('Cumulative Variance Explained')
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


