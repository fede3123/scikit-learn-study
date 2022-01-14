import pandas as pd
# Regularizes and model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
# Split data and MSE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')
    # Statistics way for see data
    print(dataset.describe())
    # Select data and is in double [[]] cuz we need axis 1
    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    # Meanwhile more expensive be alpha/Lambda more is tha penalization
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_Ridge = modelRidge.predict(X_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss: ", linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_Ridge)
    print("Ridge Loss: ", ridge_loss)

    # Coef = each column from our data set
    print("="*32)
    print("Coef LINEAR")
    print(modelLinear.coef_)

    print("="*32)
    print("Coef LASSO")
    print(modelLasso.coef_)

    print("="*32)
    print("Coef RIDGE")
    print(modelRidge.coef_)
