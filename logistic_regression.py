import numpy as np

class LogisticRegression:

    def __init__(self,n_iters=1000,lr=0.001):
        self.n_iters = n_iters
        self.lr = lr

        self.weights = None
        self.bias = None

    def fit(self,X_train,y_train):

        n_samples,n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):

            f = np.dot(X_train,self.weights) + self.bias
            y_pred = self.sigmoid(f)


            dw = (1 / n_samples) * np.dot(X_train.T, (y_pred - y_train))
            db = (1 / n_samples) * np.sum(y_pred - y_train)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def sigmoid(self,x):
        # Clip x to prevent overflow and underflow issues
        clipped_x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-clipped_x))

    def predict(self,X_test):

        f = np.dot(X_test, self.weights) + self.bias
        y_pred = self.sigmoid(f)
        y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_classes

    def get_params(self, deep=True):
        return {'lr':self.lr,
                'n_iters':self.n_iters}


# Testing
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import cross_val_score

    print(f'Accuracy:{accuracy_score(y_test, y_pred)}')

    cross_val_score = cross_val_score(lr, X, y, cv=10, scoring='accuracy')
    print(f'Average cross validation score(accuracy):{np.mean(cross_val_score)}')
    print(classification_report(y_test, y_pred))

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)