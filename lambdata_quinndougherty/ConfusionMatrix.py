import numpy as np
import pandas as pd


class ConfusionMatrix:
    '''for some 2-class classifier pipe, after running pipe.fit(X_train,
    y_train), pass into this class pipe, X_test, y_test
           to get a confusion matrix and it's associated calculations. '''

    def __init__(self, pipe, X_test, y_test):
        self.model = pipe.best_estimator_
        self.X_test = X_test
        self.y_test = y_test
        self.confusion_matrix = self.make_()

    def make_(self):
        '''This makes the confusion matrix'''
        model = self.model
        dat = self.X_test
        target = self.y_test
        seconds = ['Negative', 'Positive']
        firsts = ['Actual', 'Predicted']

        print("at an implicit decision rule of 0.5, i.e., if model(dat)>=0.5 then model(dat)=1")

        def tupdex(first):
            return [first + ' ' + seconds[0], first + ' ' + seconds[1]]

        c = np.empty((2, 2))

        def fill(i, j):
            '''mutate the matrix c with the i,j'th value'''
            val = sum([x == i and y == j for x, y in zip(
                target, model.predict(dat))])
            c[i][j] = val

        fill(0, 0)
        fill(0, 1)
        fill(1, 0)
        fill(1, 1)
        df = pd.DataFrame(c, index=tupdex(firsts[0]),
                          columns=tupdex(firsts[1]))
        return df

    def precision(self):
        '''computes precision of confusionmatrix. '''
        cm = self.confusion_matrix
        TP = cm['Predicted Positive'].loc['Actual Positive']
        PP = cm['Predicted Positive'].sum()
        return np.divide(TP, PP)

    def recall(self):
        '''computes the recall of confusionmatrix'''
        cm = self.confusion_matrix
        TP = cm['Predicted Positive'].loc['Actual Positive']
        AP = cm.loc['Actual Positive'].sum()
        return np.divide(TP, AP)

    def F1(self):
        '''computes f1 score of confusionmatrix'''
        cm = self.confusion_matrix
        prec = self.precision()
        reca = self.recall()
        return 2 * np.divide(prec * reca, prec + reca)

    def typeI(self):
        '''computes number of type1 errors'''
        cm = self.confusion_matrix
        return cm['Predicted Positive'].loc['Actual Negative']

    def confusion_report(self):
        '''reports everything about confusion matrix'''
        s1 = f'this model got a precision score of {self.precision():.3}\n'
        s2 = f'a recall score of {self.recall():.3}\n'
        s3 = f'an F1 score of {self.F1():.3}\n'
        s4 = f'and {int(self.typeI())} Type I errors'
        return s1 + s2 + s3 + s4
