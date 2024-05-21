import numpy as np

class PCA:

    def __init__(self,n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None


    def fit(self,X):
        # mean
        self.mean = np.mean(X,axis=0)
        X -= self.mean


        # covariance
        cov = np.cov(X.T)



        # eigen(vectors and values)
        eigenvalues,eigenvectors = np.linalg.eig(cov)


        # sort eigenvectors
        eignevectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]



        # store the first n eigenvectors

    def transform(self,X):

        # project data

        pass

