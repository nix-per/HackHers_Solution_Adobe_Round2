# THE SOLUTION WE HAVE PRESENTED IS PARTICULARLY FOR THE OCCLUSION PROBLEMS
# WE START BY DETECTING THE INCOMPLETE SHAPES AND THEN DRAWING THEM (REGULARIZING) TO COMPLETE THEM


import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN


# DEFINING THE COLOURS FOR PLOTTING
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# READING THE CSV FILES
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs


# WRITING PROCESSED DATA TO CSV
def write_csv(csv_path, data):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i, XYs in enumerate(data):
            for j, XY in enumerate(XYs):
                for point in XY:
                    writer.writerow([i, j] + list(point))


# VISUALISING THE CURVE
def plot(path_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()


# FUNCTION TO CLUSTER POINTS USING DBSCAN
def cluster_points(XY, eps=15, min_samples=3):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(XY)
    clusters = [XY[clustering.labels_ == i] for i in set(clustering.labels_) if i != -1]
    return clusters


# FUNCTION TO COMPLETE MISSING PARTS USING INTERPOLATION
def complete_shape(XY, num_points=100):
    t = np.linspace(0, 1, len(XY))
    x = XY[:, 0]
    y = XY[:, 1]
    t_interp = np.linspace(0, 1, num_points)
    x_interp = np.interp(t_interp, t, x)
    y_interp = np.interp(t_interp, t, y)
    return np.vstack((x_interp, y_interp)).T

#---------------------------------------------------------------------------------------------------------------------------------

# DETECTING AND REGULARIZING SHAPES
def detect_and_regularize_shapes(path_XYs):
    regularized_path_XYs = []

    for XYs in path_XYs:
        all_points = np.vstack(XYs)
        clusters = cluster_points(all_points)

        for cluster in clusters:
            completed_shape = complete_shape(cluster)
            if is_circle(completed_shape):
                regularized_path_XYs.append([regularize_circle(completed_shape)])
            elif is_ellipse(completed_shape, tolerance=10):
                regularized_path_XYs.append([regularize_ellipse(completed_shape, num_points=100)])
            elif is_rectangle(completed_shape):
                regularized_path_XYs.append([regularize_rectangle(completed_shape)])
            elif is_star(completed_shape):
                regularized_path_XYs.append([regularize_star(completed_shape)])
            elif is_straight_line(completed_shape, tolerance=1e-2):
                regularized_path_XYs.append([regularize_line(completed_shape)])
            elif is_regular_polygon(completed_shape, tolerance=10):
                regularized_path_XYs.append([regularize_polygon(completed_shape)])
            else:
                regularized_path_XYs.append([completed_shape])  # IF THE SHAPE IS NOT RECOGNISED, KEEPING IT AS IT IS

    return regularized_path_XYs

#---------------------------------------------------------------------------------------------------------------------------------

# FUNCTION TO CHECK IF A SHAPE IS A CIRCLE
def is_circle(XY):
    center = XY.mean(axis=0)
    radius = np.linalg.norm(XY - center, axis=1).mean()
    distances = np.linalg.norm(XY - center, axis=1)
    return np.allclose(distances, radius, atol=10)


# FUNCTION TO REGULARIZE A CIRCLE
def regularize_circle(XY):
    center = XY.mean(axis=0)
    radius = np.linalg.norm(XY - center, axis=1).mean()
    angles = np.linspace(0, 2 * np.pi, len(XY))
    regularized_XY = np.array([[center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)] for angle in angles])
    return regularized_XY

#---------------------------------------------------------------------------------------------------------------------------------

# FUNCTION TO CHECK IF A SHAPE IS A RECTANGLE
def is_rectangle(XY):
    if len(XY) < 4:
        return False
    hull = ConvexHull(XY)
    vertices = XY[hull.vertices]
    if len(vertices) != 4:
        return False
    return True

# FUNCTION TO REGULARIZE A RECTANGLE
def regularize_rectangle(XY):
    hull = ConvexHull(XY)
    vertices = XY[hull.vertices]
    if len(vertices) != 4:
        return XY
    min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    regularized_XY = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        [min_x, min_y]
    ])
    return regularized_XY

#---------------------------------------------------------------------------------------------------------------------------------

# FUNCTION TO CHECK IF A SHAPE IS A STAR
def is_star(XY):
    if len(XY) < 5:
        return False
    center = XY.mean(axis=0)
    distances = np.linalg.norm(XY - center, axis=1)
    return distances.std() > 10


# FUNCTION TO REGULARIZE A STAR
def regularize_star(XY):
    center = XY.mean(axis=0)
    max_radius = np.max(np.linalg.norm(XY - center, axis=1))
    num_points = 10
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = np.array([max_radius if i % 2 == 0 else max_radius / 2 for i in range(num_points)])
    regularized_XY = np.array([[center[0] + r * np.cos(angle), center[1] + r * np.sin(angle)] for r, angle in zip(radii, angles)])
    regularized_XY = np.vstack([regularized_XY, regularized_XY[0]])
    return regularized_XY

#---------------------------------------------------------------------------------------------------------------------------------

# FUNCTION FOR ELLIPSE DETECTION
def is_ellipse(XY, tolerance=20):
    def objective(params):
        a, b, cx, cy, theta = params
        if a <= 0 or b <= 0:
            return float('inf')
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        try:
            transformed_points = np.dot(XY - np.array([cx, cy]), np.array([[cos_theta / a, -sin_theta / b], [sin_theta / a, cos_theta / b]]))
            distances = np.sqrt(transformed_points[:, 0]**2 + transformed_points[:, 1]**2)
            return np.sum((distances - 1)**2)
        except:
            return float('inf')
    initial_guess = [1, 1, XY[:, 0].mean(), XY[:, 1].mean(), 0]
    result = minimize(objective, x0=initial_guess, bounds=[(0, None), (0, None), (None, None), (None, None), (None, None)])
    return result.success and result.fun < tolerance


# FUNCTION TO REGULARIZE AN ELLIPSE
def regularize_ellipse(XY, num_points=100):
    center = XY.mean(axis=0)
    u, s, vh = np.linalg.svd(XY - center)
    major_axis, minor_axis = s
    angle = np.arctan2(u[1, 0], u[0, 0])
    angles = np.linspace(0, 2 * np.pi, num_points)
    ellipse_points = np.array([
        [
            center[0] + major_axis * np.cos(a) * np.cos(angle) - minor_axis * np.sin(a) * np.sin(angle),
            center[1] + major_axis * np.cos(a) * np.sin(angle) + minor_axis * np.sin(a) * np.cos(angle)
        ]
        for a in angles
    ])
    return ellipse_points

#---------------------------------------------------------------------------------------------------------------------------------

# FUNCTION FOR POLYGON DETECTION
def is_regular_polygon(XY, tolerance=10):
    hull = ConvexHull(XY)
    vertices = XY[hull.vertices]
    if len(vertices) < 3:
        return False
    distances = np.linalg.norm(vertices - np.roll(vertices, shift=-1, axis=0), axis=1)
    return np.std(distances) < tolerance


# FUNCTION TO REGULARIZE A POLYGON
def regularize_polygon(XY):
    hull = ConvexHull(XY)
    vertices = XY[hull.vertices]
    num_sides = len(vertices)
    center = XY.mean(axis=0)
    max_radius = np.max(np.linalg.norm(vertices - center, axis=1))
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    regularized_XY = np.array([[center[0] + max_radius * np.cos(angle), center[1] + max_radius * np.sin(angle)] for angle in angles])
    return np.vstack([regularized_XY, regularized_XY[0]])

#---------------------------------------------------------------------------------------------------------------------------------

# FUNCTION FOR STRAIGHT LINE DETECTION
def is_straight_line(XY, tolerance=1e-2):
    p = np.polyfit(XY[:, 0], XY[:, 1], 1)
    distances = np.abs(np.polyval(p, XY[:, 0]) - XY[:, 1])
    return np.all(distances < tolerance)


# FUNCTION TO REGULARIZE A STRAIGHT LINE
def regularize_line(XY):
    p = np.polyfit(XY[:, 0], XY[:, 1], 1)
    x_min, x_max = np.min(XY[:, 0]), np.max(XY[:, 0])
    regularized_XY = np.array([[x_min, np.polyval(p, x_min)], [x_max, np.polyval(p, x_max)]])
    return regularized_XY

#---------------------------------------------------------------------------------------------------------------------------------

# MAIN FUNCTION TO RUN THE COMPLETE FLOW
def main():
    input_csv_path = 'problems/occlusion2.csv'    # KINDLY REPLACE THE PATH WITH ABSOLUTE PATH OF THE TEST FILES (CSV)
    output_csv_path = 'smoothed_shapes.csv'       # KINDLY REPLACE THE PATH WITH ABSOLUTE PATH OF THE OUTPUT FILE (CSV)

    path_XYs = read_csv(input_csv_path)
    regularized_path_XYs = detect_and_regularize_shapes(path_XYs)
    plot(regularized_path_XYs)
    write_csv(output_csv_path, regularized_path_XYs)
    
    return regularized_path_XYs

if __name__ == "__main__":
    main()
