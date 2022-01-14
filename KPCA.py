import pandas as pd
import matplotlib.pyplot as plt
# Data preprocessing
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Model
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')

    print(dt_heart.head(5))
    # Sing features and target
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    # Data normalization before PCA
    dt_features = StandardScaler().fit_transform(dt_features)
    # Split data
    # random_state is for make sure that always is tha same split
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=45)

    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(dt_train, y_train)
    print("Score KPCA: ", logistic.score(dt_test, y_test))
