import numpy as np
import matplotlib.pyplot as plt
import pylab

import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit as ss

from sklearn.metrics import classification_report

from xgboost import XGBClassifier
from xgboost import plot_importance


def select_best_features(features, target,):
    x_train, x_test, y_train, y_test = train_test_split(features, 
                                                        target,
                                                        test_size=.25,
                                                        random_state=21)
    classifier = XGBClassifier()
    xgb_model = classifier.fit(x_train, y_train)
    xgb_predictions = classifier.predict(x_test)
    report = classification_report(y_test, xgb_predictions)
    print(f'Classification report: {report}')
    p = plot_importance(xgb_model)
    plt.show()
    return


def check_norm_dist(features, target, params,):
    rs = ss(n_splits=150, train_size=.7, random_state=0)
    scores_0 = []
    scores_1 = []
    mean_0 = []
    mean_1 = []
    S = []
    
    for train_index, _ in rs.split(features):
        classifier0 = XGBClassifier(n_estimators=params[0], max_depth=4, silent=1)
        score0 = cross_val_score(classifier0, features.iloc[train_index], target[train_index], scoring='roc_auc', cv=10)
        scores_0.append(score0)
        
        classifier1 = XGBClassifier(n_estimators=params[1], max_depth=4, silent=1)
        score1 = cross_val_score(classifier1, features.iloc[train_index], target[train_index], scoring='roc_auc', cv=10)
        scores_1.append(score1)
        
    scores_0 = np.asmatrix(scores_0)
    scores_1 = np.asmatrix(scores_1)
    
    for i in range(scores_0.shape[0]):
        diff = []
        mean_0.append(scores_0[i].mean())
        mean_1.append(scores_1[i].mean())
        for j in range(scores_0.shape[1]):
            diff.append(scores_0[i, j] - scores_1[i, j])
        S.append(np.array(diff).var())    
    means = list((map(lambda x, y: x - y, mean_0, mean_1)))
    
    pylab.subplot(1, 2, 1)
    stats.probplot(S, dist="chi2", sparams=(150), plot=pylab)

    pylab.subplot(1, 2, 2)
    stats.probplot(means, dist="norm", plot=pylab)
    pylab.show()

    return means, S


def select_best_estimator(features, target,):
    trees = [] + list(range(10, 100, 10))
    xgb_scoring = []
    means = []
    for tree in trees:
        classifier = XGBClassifier(n_estimators=tree, max_depth=4, silent=1)
        score = cross_val_score(classifier, features, target, scoring='roc_auc', cv=5)
        xgb_scoring.append(score)
    xgb_scoring = np.asmatrix(xgb_scoring)

    for i in range(xgb_scoring.shape[0]):
        means.append(xgb_scoring[i].mean())
        
    plt.plot(trees, means)


def make_model(train_data, target, n_estimators,):
    params = {'objective': 'binary:logistic', 
              'max_depth': 4, 
              'silent': 1}
    n_estimators = n_estimators

    classifier = XGBClassifier(n_estimators=n_estimators, params=params)
    xgb_model = classifier.fit(train_data, target)

    return xgb_model