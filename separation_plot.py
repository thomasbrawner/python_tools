## separation_plot.py
## Separation plots for visualizing classification performance for binary dependent variables
## Citation: Greenhill, Ward, and Sacks (2011)
## Link to paper: http://tinyurl.com/qc52uf4 
## tb 29 jun 2015, last update 24 oct 2015

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 


def separation_plot(y_true, y_pred, alpha=0.80, fname=None, show=False):
    '''
    Display observed events against predicted probabilities. 

    Arguments
    ---------
        y_true : vector of observed class labels
        y_pred : vector of predicted probabilities
        alpha: float from 0 to 1, transparency
        fname: file path to directory to save plot 
        show: boolean, whether to display the plot 
    '''
    pdata = pd.DataFrame([y_true, y_pred]).T
    pdata.columns = ['y', 'yhat']
    pdata = pdata.sort('yhat')
    pdata = pdata.reset_index(drop=True)

    events = pdata[pdata['y'] == 1]
    evals = events.index.values

    plt.figure(figsize=(12, 2.5))
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax = plt.gca()
    ax.set_xlim([(evals[0] - 1), (evals[-1] + 1)])
    ax.set_ylim([0, 1])
    ax.plot(pdata['yhat'], '-')
    for i in evals:
        ax.axvline(x=i, linewidth=0.15, linestyle='-', color='r', alpha=alpha)

    if show:
        plt.show() 

    if fname is not None:
        plt.savefig(fname)
        plt.close() 

