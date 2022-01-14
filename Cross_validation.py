import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')

    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    model = DecisionTreeRegressor()
    #                                   how many cross         What operation
    score = cross_val_score(model, X,y, cv =3, scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))

    #           in dataset split train and test in 3 times
    kf = KFold(n_splits=3, shuffle=True, random_state=43)

    for train, test in kf.split(dataset):
        print(train)
        print(test)
