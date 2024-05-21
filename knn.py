
import numpy as np
from collections import Counter



def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))




def most_common_element(k_neighbours):

    c = 0
    common_element = 0

    maximum_count = 0

    for i in k_neighbours:
        c = k_neighbours.count(i)

        if maximum_count < c:
            maximum_count = c
            common_element = i

    return common_element

# print(most_common_element([1,2,1,3,5,7]))
# print(Counter([1,2,1,3,5,7]))
#
#
# print(Counter([1,2,1,3,5,7]).most_common(1)[0][0])

class KNN:
    def __init__(self,k):
        self.k = k


    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X_test):
        predicted_labels = [self._predict(x) for x in X_test]
        return np.array(predicted_labels)

    def _predict(self,x):
        # compute the distances all sample points from x
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]

        # get k nearest neighbours of x
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]


        # get the majority vote for the most common class label
        return most_common_element(k_labels)

    # def score(self, X, y):
    #     y_pred = self.predict(X)
    #     correct_predictions = np.sum(y == y_pred)
    #     total_predictions = len(y)
    #     accuracy = correct_predictions / total_predictions
    #     return accuracy

    def get_params(self, deep=True):
        return {'k': self.k}

    # def set_params(self, **params):
    #     self.k = params['k']
    #     return self

"Note...." \
"When you use cross_val_score, it internally uses clone from scikit-learn to " \
"create copies of the estimator for each fold of cross-validation. The get_params method is called " \
"during this cloning process to ensure that each copy of the estimator has the same hyperparameter settings."

"In summary, while you may not explicitly call get_params in your code for cross_val_score, " \
"scikit-learn uses it behind the scenes to create consistent copies of your estimator during cross-validation and" \
" hyperparameter tuning processes. This consistency is important for the reproducibility and reliability of your machine learning " \
"workflow."

# Testing

if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn import datasets


    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



    clf = KNN(3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import cross_val_score

    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Assuming clf is your trained model and X, y are your features and labels
    cross_val_scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

    # Print or use cross_val_scores as needed
    print(cross_val_scores)
    print("Avg. cross val score:", np.mean(cross_val_scores))