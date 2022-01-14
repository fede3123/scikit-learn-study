import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':
    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(10))

    X = dataset.drop(['competitorname'], axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=10).fit(X)
    print("Total de centros: ", len(kmeans.cluster_centers_))
    print("="*35)
    # this array is in order according our data
    print(kmeans.predict(X))

    # Create a new column for groups
    dataset['group'] = kmeans.predict(X)

    print("="*35)
    print(dataset)
