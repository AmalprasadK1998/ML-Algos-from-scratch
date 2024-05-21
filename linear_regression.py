import numpy as np


class LinearRegression:

    def __init__(self,lr = 0.001,n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters

        self.weights = None
        self.bias = None

    def fit(self,X_train,y_train):

        # Weight initialisation
        n_samples,n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0





        for i in range(self.n_iters):

            y_pred = np.dot(X_train, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X_train.T, (y_pred - y_train))
            db = (1 / n_samples) * np.sum(y_pred - y_train)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db




    def predict(self,X_test):
        return np.dot(X_test,self.weights) + self.bias

    def get_params(self, deep=True):
        return {'lr':self.lr,
                'n_iters':self.n_iters}


# Testing
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X.shape)
    # print(y.shape)

    # fig = plt.figure(figsize=(8,6))
    # plt.scatter(X[:,0],y,color="r",marker="o",s=30)
    # plt.show()



    lr = LinearRegression(lr=0.05)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, r2_score

    mse = np.mean((y_test - y_pred) ** 2)
    print(f"Mean Squared Error:{mse}")
    print(f"r2 score:{r2_score(y_test, y_pred)}")
    cross_val_score = cross_val_score(lr, X, y, cv=10, scoring='r2')
    print(f'Mean Cross Validation Score(r2):{np.mean(cross_val_score)}')

    line = lr.predict(X)
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test, label='test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(X, line, color='black', label='prediction')
    plt.show()



