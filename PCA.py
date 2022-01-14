import pandas as pd
import matplotlib.pyplot as plt
# Data preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Model
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
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

    print(X_train.shape)
    print(y_train.shape)

    # Components default its equal to min(n_samples, n_features)
    # Components = the higher importance and variance
    pca = PCA(n_components=3)
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    # X axis                                        Y axis
    # X = components, Y = importance
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    # params for sklearn
    logistic = LogisticRegression(solver='lbfgs')

    # Do PCA to dt_train and dt_test, later fit the model and print graphics
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("Score PCA: ", logistic.score(dt_test, y_test))

    # Do same but with IPCA
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("Score IPCA: ", logistic.score(dt_test, y_test))