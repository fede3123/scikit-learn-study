import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')

    print(dataset)

    X = dataset.drop(['country', 'rank', 'score'], axis=1)
    y = dataset['score']
    reg = RandomForestRegressor()

    parametros = {
        'n_estimators': range(4,16),
        'criterion': ['mse', 'mae'],
        'max_depth': range(2,11)
    }
    # ten different combinations for parameters and take all data and split in 3 part 2 training and 1 for test
    rand_est = RandomizedSearchCV(reg, parametros, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X,y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[0]]))