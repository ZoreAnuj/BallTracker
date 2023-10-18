import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objs as go


def TLS(a,b,c):
    A = np.column_stack((a, b, c))
    A_mean = np.mean(A, axis=0)
    A_centered = A - A_mean
    cov = np.dot(A_centered.T, A_centered)
    eigvals, eigvecs = np.linalg.eig(cov)
    n = eigvecs[:, np.argmin(eigvals)]
    a, b, c = n[:3]
    d = -(a*A_mean[0] + b*A_mean[1] + c*A_mean[2])

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    x, y = np.meshgrid(x, y)
    z = (-a/c) * x + (-b/c) * y + (-d/c)

    return x, y, z




if __name__ == '__main__':
    df1 = pd.read_csv("pc1.csv", sep=',', header=None)  # ,error_bad_lines=False,
    df2 = pd.read_csv("pc2.csv", sep=',', header=None)  # ,error_bad_lines=False,

    print("Covariance for PC1: ")

    data1 = df1
    N = data1.shape[0]
    n_dim = data1.shape[1]
    cov = np.zeros((n_dim, n_dim))

    data_arr1 = data1.to_numpy()

    m, n = data_arr1.shape
    covar = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            avg1 = (sum(data_arr1[:, i])) / m
            avg2 = (sum(data_arr1[:, j])) / m
            var = 0
            for k in range(m):
                var += (data_arr1[k, i] - avg1) * (data_arr1[k, j] - avg2)
            covar[i][j] = var / m

    #print('numpy cov', np.cov(data1, rowvar=False, bias=False))

    print(covar)

    import numpy as np

    points1 = data_arr1

    import numpy as np

    # Calculate the centroid of the points
    centroid = np.mean(points1, axis=0)

    # Compute the covariance matrix of the points
    covariance = np.zeros((3, 3))
    for p in points1:
        diff = p - centroid
        covariance += np.outer(diff, diff)
    covariance /= len(points1)

    # Find the eigenvalues and eigenvectors of the covariance matrix
    eig_values, eig_vectors = np.linalg.eig(covariance)

    # Choose the eigenvector with the smallest eigenvalue as the surface normal
    surface_normal = eig_vectors[:, np.argmin(eig_values)]

    # Calculate the surface magnitude
    surface_magnitude = np.sqrt(np.sum(surface_normal ** 2))

    # Calculate the surface direction
    surface_direction = surface_normal / surface_magnitude

    print("Eigenvalues:", eig_values)
    print("Surface magnitude:", surface_magnitude)
    print("Surface direction:", surface_direction)


    #Plot

    PC1 = plt.figure(figsize=(12, 12))
    ax = PC1.add_subplot(projection='3d')
    ax.scatter(data_arr1[:, 0], data_arr1[:, 1], data_arr1[:, 2], marker="o", c="red")
    plt.title("Raw PC1 Plot")
    plt.show()

    data_arr2 = df2.to_numpy()
    figure = plt.figure(figsize=(12, 12))
    ax = figure.add_subplot(projection='3d')
    ax.scatter(data_arr2[:, 0], data_arr2[:, 1], data_arr2[:, 2], marker="o", c="red")
    plt.title("Raw PC2 Plot")
    plt.show()

    #Standard Least Square

    dfr1 = pd.read_csv("pc1.csv", header=None)
    dfr1 = np.array(dfr1)
    dfr1 = np.transpose(dfr1)

    a1 = dfr1[0]
    b1 = dfr1[1]
    c1 = dfr1[2]

    dfr2 = pd.read_csv("pc2.csv", header=None)
    dfr2 = np.array(dfr2)
    dfr2 = np.transpose(dfr2)

    a2 = dfr2[0]
    b2 = dfr2[1]
    c2 = dfr2[2]

    # Total Least Square

    x1, y1, z1 = TLS(a1,b1,c1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=a1,
            y=b1,
            z=c1,
            mode='markers',
            marker=dict(
                size=2,
                color="green",
                colorscale='Viridis',
                opacity=0.8
            )
        )
    )

    fig.add_trace(
        go.Surface(
            x=x1,
            y=y1,
            z=z1,
            colorscale='YlOrRd',
            opacity=0.8
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    fig.show()

    x2, y2, z2 = Total_Least_Square(a2, b2, c2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=a2,
            y=b2,
            z=c2,
            mode='markers',
            marker=dict(
                size=2,
                color="green",
                colorscale='Viridis',
                opacity=0.8
            )
        )
    )

    fig.add_trace(
        go.Surface(
            x=x2,
            y=y2,
            z=z2,
            colorscale='YlOrRd',
            opacity=0.8
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    fig.show()






