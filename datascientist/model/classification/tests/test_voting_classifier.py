from datascientist.model.classification.skl.ensemble_models.voting_classifier import _voting_classifier

import numpy as np
from pytest import raises

def test_voting_classifier():

    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2]))

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2]))

    metrics = 'accuracy'
    answer = _voting_classifier(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'voting classifier'
    assert answer[1] == 1.0
    assert answer[2] is None

    metrics = 'f1'
    answer = _voting_classifier(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'voting classifier'
    assert answer[1] == np.array([1., 1., 1., 1.])
    assert answer[2] is None

    metrics = 'jaccard'
    answer = _voting_classifier(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'voting classifier'
    assert answer[1] == np.array([1., 1., 1., 1.])
    assert answer[2] is None