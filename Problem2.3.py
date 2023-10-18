import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objs as go

# Load the data from the CSV file
data = pd.read_csv('pc1.csv').values

# Define the fit_plane function for RANSAC
def fit_plane(data, threshold):
    # Select three random points from the data
    indices = np.random.choice(data.shape[0], size=3, replace=False)
    p1, p2, p3 = data[indices, :]

    # Compute the normal vector to the plane defined by the three points
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)

    # Compute the distance from the origin to the plane along the normal vector
    d = -np.dot(n, p1)

    # Compute the inlier mask for the plane
    distances = np.abs(np.dot(data[:, :3], n) + d)
    inlier_mask = distances < threshold

    # Return the number of inliers and the plane coefficients
    return np.sum(inlier_mask), np.concatenate((n, [d]))


if __name__ == '__main__':
    # Run RANSAC to find the best plane
    best_plane = None
    best_inliers = 0
    for i in range(1000000):
        num_inliers, plane = fit_plane(data, 0.01)
        if num_inliers > best_inliers:
            best_plane = plane
            best_inliers = num_inliers

    # Extract the plane coefficients
    a, b, c, d = best_plane

    # Create a meshgrid to represent the plane
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    zz = (-a * xx - b * yy - d) / c

    # Compute the inlier and outlier masks
    distances = np.abs(np.dot(data[:, :3], [a, b, c]) + d)
    inlier_mask = distances < 0.1
    outlier_mask = np.logical_not(inlier_mask)
    inliers = data[inlier_mask, :]
    outliers = data[outlier_mask, :]

    # Create a trace for the inliers
    inlier_trace = go.Scatter3d(
        x=inliers[:, 0],
        y=inliers[:, 1],
        z=inliers[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
        ),
        name='Inliers'
    )

    # Create a trace for the outliers
    outlier_trace = go.Scatter3d(
        x=outliers[:, 0],
        y=outliers[:, 1],
        z=outliers[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
        ),
        name='Outliers'
    )

    # Create a trace for the plane
    plane_trace = go.Surface(
        x=xx,
        y=yy,
        z=zz,
        colorscale='Greens',
        opacity=0.8,
        showscale=False,
        name='Plane'
    )

    # Create the figure and add the traces
    fig = go.Figure([inlier_trace, outlier_trace, plane_trace])

    # Show the plot
    fig.show()

    data = pd.read_csv('pc2.csv').values

    best_plane = None
    best_inliers = 0
    for i in range(1000000):
        num_inliers, plane = fit_plane(data, 0.01)
        if num_inliers > best_inliers:
            best_plane = plane
            best_inliers = num_inliers

    # Extract the plane coefficients
    a, b, c, d = best_plane

    # Create a meshgrid to represent the plane
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    zz = (-a * xx - b * yy - d) / c

    # Compute the inlier and outlier masks
    distances = np.abs(np.dot(data[:, :3], [a, b, c]) + d)
    inlier_mask = distances < 0.1
    outlier_mask = np.logical_not(inlier_mask)
    inliers = data[inlier_mask, :]
    outliers = data[outlier_mask, :]

    # Create a trace for the inliers
    inlier_trace = go.Scatter3d(
        x=inliers[:, 0],
        y=inliers[:, 1],
        z=inliers[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
        ),
        name='Inliers'
    )

    # Create a trace for the outliers
    outlier_trace = go.Scatter3d(
        x=outliers[:, 0],
        y=outliers[:, 1],
        z=outliers[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
        ),
        name='Outliers'
    )

    # Create a trace for the plane
    plane_trace = go.Surface(
        x=xx,
        y=yy,
        z=zz,
        colorscale='Greens',
        opacity=0.8,
        showscale=False,
        name='Plane'
    )

    # Create the figure and add the traces
    fig = go.Figure([inlier_trace, outlier_trace, plane_trace])

    # Show the plot
    fig.show()