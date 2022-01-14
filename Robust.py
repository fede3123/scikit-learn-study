import pandas as pd

from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    X = dataset.drop(['country', 'score', 'rank'], axis=1)
    y = dataset[['score']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

    estimators = {
        #                          Penalization, if c is small +Bias and -variance
        'SVR': SVR(gamma= 'auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    for key, value in estimators.items():
        value.fit(X_train, y_train)
        predictions = value.predict(X_test)

        print("="*35)
        print(key)
        print("MSE : ", mean_squared_error(y_test, predictions))