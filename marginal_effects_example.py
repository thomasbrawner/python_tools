import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import sys
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.utils import resample
sns.set_style('white')


def dummify(df, variable): 
    '''
    Generate dummy matrix from categorical variable, merge into 
    data frame, then drop the variable and return the new data frame. 
    '''
    dummies = pd.get_dummies(df[variable], prefix=variable)
    output = df.join(dummies)
    return output.drop(variable, axis=1)
    
def generate_data(): 
    '''
    Read and merge GWF data and control variables.
    Subset the data frame and format for analysis. 
    '''
    data = pd.read_table('regime_failure/gwf.txt', sep=',')
    data = pd.merge(data, pd.read_table('regime_failure/control_variables.txt', sep=','), on=['cowcode', 'year'], how='left')
    data['period'] = data['year'] // 2 * 2
    data['growth'] /= 10
    data = data[['failure', 'gdppc', 'growth', 'resource', 'population', 'period', 'region']]
    data = dummify(data, 'region')
    data = dummify(data, 'period')
    return data.dropna()  

def optimal_l2(X, y): 
    '''
    Find the optimal level of L2 regularization for logistic regression
    '''
    logit = LogisticRegressionCV(Cs=50, cv=10)
    logit.fit(X, y)
    return logit.C_

def boot_estimates(model, X, y, nboot):
    '''
    Evaluate coefficient estimates for nboot boostrap samples
    '''
    coefs = [np.hstack([model.fit(iX, iy).intercept_, model.fit(iX, iy).coef_.ravel()]) 
            for iX, iy in (resample(X, y) for i in xrange(nboot))]  
    return np.vstack(coefs)

def inverse_logit(X, estimates): 
    '''
    Logistic transformation of linear predictor
    '''
    return 1 / (1 + np.exp(-X.dot(estimates.T)))

def predicted_probability(X, estimates): 
    '''
    Evaluate the mean predicted probability across bootstrap estimates 
    and observations in X. 
    '''
    probabilities = inverse_logit(X, estimates)
    mean_probability_by_obs = probabilities.mean(axis=1)
    return probabilities.mean() 
    

if __name__ == '__main__': 

    # set up the data
    data = generate_data() 
    y = data.pop('failure')
    X = data.values 

    # obtain bootstrap estimates 
    logit = LogisticRegression(C=optimal_l2(X, y))
    estimates = boot_estimates(logit, X, y, nboot=500)

    # evaluate mean predicted probability at each counterfactual value of growth
    growth_values = np.linspace(data['growth'].min(), data['growth'].max())
    growth_loc = data.columns.tolist().index('growth')
    probabilities = [] 
    Xc = X.copy()
    Xc = np.insert(Xc, 0, 1, axis=1)
    for value in growth_values: 
        Xc[:, growth_loc] = value
        probabilities.append(predicted_probability(Xc, estimates))
    
    # plot marginal effects superimposed on histogram of growth
    fig, ax1 = plt.subplots() 
    ax1.plot(growth_values, probabilities, color='k', linestyle='-')
    ax1.set_xlabel('Lagged 2-Year Moving Average of Growth', labelpad=11)
    ax1.set_ylabel('Predicted Probability of Failure', labelpad=11)
    ax1.set_ylim([0.0, 1.0])

    ax2 = ax1.twinx() 
    ax2.hist(X[:, growth_loc], bins=20, color='k', alpha=0.35)
    ax2.set_ylabel('Frequency')

    plt.tight_layout() 
    plt.show() 
 
