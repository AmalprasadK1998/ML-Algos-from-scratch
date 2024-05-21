import numpy as np


class SVM:
    def __init__(self, lr=0.001, lam=0.01, n_iters=1000):
        # Hyperparameters
        # lr-learning rate,lp-lambda,n_ters-iterations
        self.lr = lr
        self.lp = lam
        self.iterations = n_iters

        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y > 0, 1, -1)
        n_samples, n_features = X.shape

        # Initialization of parameters (weights and bias)
        self.w = np.zeros(n_features)
        self.b = 0
        
        # optimization using Gradient Descent
        for _ in range(self.iterations):
            for i, x_i in enumerate(X):

                # if y*f(x) >= 1...every data point is classified correctly by the model
                condition = y[i] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    dw = 2 * self.lp * self.w
                    self.w -= self.lr * dw

                    # db = 0
                    # Therefore self.b = self.b(bias remains unchanged)

                else:
                    dw = 2 * self.lp * self.w - np.dot(x_i, y_[i])
                    db = y_[i]
                    self.w -= self.lr * dw
                    self.b -= self.lr * db

    def predict(self, X):
        y = [self._predict(x) for x in X]
        return y

    def _predict(self, x):
        y = np.dot(self.w, x)
        return 1 if y >= 0 else -1

    def get_params(self, deep=True):
        return {'lr': self.lr,
                'lam': self.lp,
                'n_iters': self.iterations}


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split, cross_val_score
    import matplotlib.pyplot as plt

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    y = np.where(y <= 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    clf = SVM()
    clf.fit(X, y)
    predictions = clf.predict(X_test)

    # print(clf.w, clf.b)
    print('Accuracy:', accuracy_score(y_test, predictions))
    cvs = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print(cvs)
    print('Avg cross val score:', np.mean(cvs))
    print(classification_report(y_test, predictions))




    # NOTES....

    # Data leak:
    """Using a different type of data for classification won't necessarily solve the potential data leakage issue.
    The problem arises from the fact that the data splitting operation should be done before any preprocessing or 
    modeling steps to avoid unintentional information leakage from the test set into the training set."""


    # Hyperparameters:
    """Definition: Hyperparameters are external configuration settings that
    are set before training the model.They are not learned from the data but are
    specified by the machine learning engineer or data scientist."""


    # parameters:
    """Definition: Parameters are the internal variables that the model learns from the training data. 
    They are the coefficients or weights associated with input features."""