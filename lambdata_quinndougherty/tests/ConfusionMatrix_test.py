#!/usr/bin/env python

import numpy as np
from functools import reduce
from numpy.testing import assert_almost_equal
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import unittest
from ConfusionMatrix import *


dat_url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data'

'''A retrospective sample of males in a heart-disease high-risk region
of the Western Cape, South Africa. There are roughly two controls per
case of CHD. Many of the CHD positive men have undergone blood
pressure reduction treatment and other programs to reduce their risk
factors after their CHD event. In some cases the measurements were
made after these treatments. These data are taken from a larger
dataset, described in  Rousseauw et al, 1983, South African Medical
Journal.

sbp		systolic blood pressure
tobacco		cumulative tobacco (kg)
ldl		low densiity lipoprotein cholesterol
adiposity
famhist		family history of heart disease (Present, Absent)
typea		type-A behavior
obesity
alcohol		current alcohol consumption
age		age at onset
chd		response, coronary heart disease
'''


df = pd.read_csv(dat_url).drop('row.names', axis=1)

assert all([x == 0 for x in df.isna().sum().values])

X = df.drop(['chd', 'famhist'], axis=1)
X['famhist'] = df.famhist.map({'Present': 1, 'Absent': 0})
y = df.chd

X_train, X_test, y_train, y_test = train_test_split(X, y)

pipe = Pipeline(steps=[
    ('scale', StandardScaler()),
    ('classi', LogisticRegression())
])

paramgrid = {'scale__with_std': ['False', 'True'],
             'classi__C': [np.exp(k) for k in range(-4, 3, 2)],
             'classi__penalty': ['l1', 'l2']}

gs = GridSearchCV(pipe, param_grid=paramgrid, cv=9, n_jobs=-1)
gs.fit(X_train, y_train)


class ConfMatTests(unittest.TestCase):
    def confmat_text(self):
        assertEquals(ConfusionMatrix, X_test, y_test).confusion_matrix.shape, (2, 2))

if __name__ == '__main__':
    unittest.main()
