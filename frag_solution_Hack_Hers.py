# THE SOLUTION WE HAVE PRESENTED IS PARTICULARLY FOR THE "FRAG" PROBLEMS
# WE START BY DETECTING THE FRAGMENTED SHAPES AND THEN DRAWING (REGULARIZING) THEM 


import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy import optimize


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
            ax.scatter(XY[:, 0], XY[:, 1], c=c)
    ax.set_aspect('equal')
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------------

# DETECTING AND REGULARIZING SHAPES
def detect_and_regularize_shapes(path_XYs):
    regularized_path_XYs = []

    for XYs in path_XYs:
        for XY in XYs:
            if is_line(XY):
                regularized_path_XYs.append([regularize_line(XY)])
            elif is_circle(XY):
                regularized_path_XYs.append([regularize_circle(XY)])
            elif is_rectangle(XY):
                regularized_path_XYs.append([regularize_rectangle(XY)])
            elif is_star(XY):
                regularized_path_XYs.append([regularize_star(XY)])
            else:
                regularized_path_XYs.append([XY])  # If not recognized, keeping the original

    return regularized_path_XYs

#---------------------------------------------------------------------------------------------------------------------------------

# FUNCTION TO CHECK IF A SHAPE IS A CIRCLE
def is_circle(XY):
    center = XY.mean(axis=0)
    radius = np.linalg.norm(XY - center, axis=1).mean()
    distances = np.linalg.norm(XY - center, axis=1)
    return np.allclose(distances, radius, atol=10)  # Threshold for determining circularity


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
    
    # Calculating the convex hull to get vertices in order
    hull = ConvexHull(XY)
    vertices = XY[hull.vertices]
    
    # Ensuring there are approximately 4 vertices
    if len(vertices) < 4:
        return False
    
    # Function to compute the distance between two points
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    # Function to compute the angle between two vectors
    def angle_between(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # Calculating edge lengths and angles between edges
    def is_perpendicular(v1, v2):
        return np.isclose(angle_between(v1, v2), np.pi / 2, atol=0.2)  
    
    distances = [distance(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
    
    # Checking if there are exactly 4 vertices
    if len(vertices) == 4:
        edges = [vertices[i] - vertices[(i + 1) % 4] for i in range(4)]
        
        # Checking if all angles are approximately 90 degrees
        if all(is_perpendicular(edges[i], edges[(i + 1) % 4]) for i in range(4)):
            return True
        
    # If not exactly 4 vertices, checking if there are approximately 4 vertices
    if len(vertices) >= 4:
        # Checking if distances form a reasonable rectangle (2 pairs of equal lengths)
        if np.isclose(distances[0], distances[2], atol=5) and np.isclose(distances[1], distances[3], atol=5):
            return True
    
    return False


# FUNCTION TO REGULARIZE A RECTANGLE
def regularize_rectangle(XY):
    # Calculating the convex hull to get vertices in order
    hull = ConvexHull(XY)
    vertices = XY[hull.vertices]
    
    # Ensuring there are approximately 4 vertices
    if len(vertices) < 4:
        return XY  # Return the original if not enough vertices
    
    # Calculating the bounding box of the rectangle
    min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    
    # Computing the center, width, and height of the bounding box
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    width = max_x - min_x
    height = max_y - min_y
    
    # Creating a regular rectangle with the same width and height
    regularized_XY = np.array([
        [center[0] - width / 2, center[1] - height / 2],
        [center[0] + width / 2, center[1] - height / 2],
        [center[0] + width / 2, center[1] + height / 2],
        [center[0] - width / 2, center[1] + height / 2],
        [center[0] - width / 2, center[1] - height / 2]  # Close the rectangle
    ])
    
    return regularized_XY

#---------------------------------------------------------------------------------------------------------------------------------

# FUNCTION TO CHECK IF A SHAPE IS A STAR
def is_star(XY):
    # Basic heuristic (logic used): a star has more than 4 points and a high variance in distances from the center
    if len(XY) < 5:
        return False
    center = XY.mean(axis=0)
    distances = np.linalg.norm(XY - center, axis=1)
    return distances.std() > 10  # Threshold for determining starriness


# FUNCTION TO REGULARIZE A STAR
def regularize_star(XY):
    # Calculating the center of the star
    center = XY.mean(axis=0)
    
    # Calculating the maximum distance from the center to any point (to determine the size)
    max_radius = np.max(np.linalg.norm(XY - center, axis=1))
    
    # Defining the number of points and alternate radii
    num_points = 10
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = np.array([max_radius if i % 2 == 0 else max_radius / 2 for i in range(num_points)])
    
    # Creating the star points with disconnected last edge
    regularized_XY = np.array([[center[0] + r * np.cos(angle), center[1] + r * np.sin(angle)] for r, angle in zip(radii, angles)])
    
    # Ensuring the last point connects back to the first point
    regularized_XY = np.vstack([regularized_XY, regularized_XY[0]])
    
    return regularized_XY

#---------------------------------------------------------------------------------------------------------------------------------

# FUNCTION TO CHECK IF THE SHAPE IS A LINE
def is_line(XY):
    if len(XY) < 2:
        return False
    # Fitting a line to the points and check if most points are close to this line
    def line_func(params, x):
        return params[0] * x + params[1]

    def error_func(params, x, y):
        return y - line_func(params, x)

    x = XY[:, 0]
    y = XY[:, 1]
    params, _ = optimize.leastsq(error_func, [1, 0], args=(x, y))
    fitted_y = line_func(params, x)
    
    # Checking if the deviation from the fitted line is within a lenient tolerance
    deviations = np.abs(y - fitted_y)
    tolerance = 20  
    num_close_points = np.sum(deviations < tolerance)
    
    # Considering it a line if more than a certain percentage of points are close to the fitted line
    return num_close_points / len(XY) > 0.8  


# FUNCTION TO REGULARIZE THE LINE
def regularize_line(XY):
    # Fitting a line to the points and generate a regularized line with start and end points
    def line_func(params, x):
        return params[0] * x + params[1]

    def error_func(params, x, y):
        return y - line_func(params, x)

    x = XY[:, 0]
    y = XY[:, 1]
    params, _ = optimize.leastsq(error_func, [1, 0], args=(x, y))
    fitted_y = line_func(params, x)
    
    start_point = np.array([x[0], fitted_y[0]])
    end_point = np.array([x[-1], fitted_y[-1]])
    
    return np.array([start_point, end_point])

#---------------------------------------------------------------------------------------------------------------------------------

# READING, PROCESSING, PLOTTING, AND WRITING DATA
input_csv_path = 'problems/frag2.csv'             # KINDLY REPLACE THE PATH WITH ABSOLUTE PATH OF THE TEST FILES (CSV)
output_csv_path = 'smoothed_shapes.csv'           # KINDLY REPLACE THE PATH WITH ABSOLUTE PATH OF THE OUTPUT FILE (CSV)

path_XYs = read_csv(input_csv_path)
processed_path_XYs = detect_and_regularize_shapes(path_XYs)
plot(processed_path_XYs)
write_csv(output_csv_path, processed_path_XYs)


