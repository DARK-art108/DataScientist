from sklearn.ensemble import VotingClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
import numpy as np

def _voting_classifier(train,test,estimators, *,x_predict=None, voting='hard',metrics=None, weights=None, n_jobs=None, flatten_transform=True, verbose=False):
    
    """
    For more info visit : 
    
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
    
    """

    model = VotingClassifier(estimators,  voting ='hard')
    model.fit(train[0], train[1])
    model_name = 'Voting_Classifier'
    y_hat = model.predict(test[0])

    if metrics == 'accuracy':
        accuracy = accuracy_score(test[1], y_hat)

    if metrics == 'f1':
        accuracy = f1_score(test[1], y_hat)

    if metrics == 'jaccard':
        accuracy = jaccard_score(test[1], y_hat)


    if x_predict is None:
        return (model_name, accuracy, None)

    y_predict = model.predict(x_predict)
    return (model_name, accuracy, y_predict)

