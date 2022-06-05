"""
Utility functions for label generation.
"""

import numpy as np
import pandas as pd
from pygeoif import from_wkt
from scipy.stats import multivariate_normal
from skimage.morphology import binary_dilation, disk

from utils.conversion_utils import process_point


def create_label(
    annotations: pd.core.frame.DataFrame, 
    corner_points: tuple, 
    processing_info: tuple,
    width: int,
    height: int,
    scale: float,
    diameter: float, 
    output_shape: tuple,
    gaussian: bool = False
) -> np.ndarray:
    """ 
    Args:
        annotations:  collection of all annotations for one specific frame.
        corner_points:  four annotated corner points.
        processing_info:  information about how the image was processed to correct the annotated points.
        width:  labelmap width in pixels.
        height:  labelmap height in pixels.
        scale:  value indicating the pixel/cm ratio.
        diameter:  value indicating the desired diameter of the circular structuring element in cm.
        output_shape:  shape of output labelmap.
        gaussian:  if True, gaussians are placed on the annotated positions.
                   if False, full intensity circles are placed on the annotated positions.
    Returns:
        label:  cartesian line labelmap created based on the annotations.
    """ 
    # create an empty image to add the labels to
    label = np.zeros(output_shape, dtype=np.uint8)
    
    # calculate the radius
    radius = diameter / 2 * scale * processing_info[2]

    lines = []
    for annotation in eval(annotations.wkt_geoms.tolist()[0]):
        # get the annotated points that form a line
        line = from_wkt(annotation).coords 
        # rescale the coordinates from the 0-100 range to the image-specific ranges
        line = [(line[i][0]/100*width, line[i][1]/100*height) for i in np.arange(len(line))]
        # process the coordinates to correct for centering, padding, and rescaling
        lines.append(tuple([process_point(point, processing_info) for point in line]))

    # get the top points
    top_points = [get_top_point(line) for line in lines]
    # create a labelmap
    if gaussian:
        label = create_gaussian_label(top_points, radius, output_shape)
    else:
        label = create_point_label(top_points, round(radius), output_shape)

    return label, radius

def get_top_point(points: list) -> tuple:
    """
    Args:
        points:  all annotated points where each point is of the shape (x, y).

    Returns:
        point:  top point as (x, y)
    """
    top_point = None
    for point in points:
        if top_point == None:
            top_point = point
        elif point[1] < top_point[1]:
            top_point = point
    
    return top_point

def create_point_label(top_points: list, radius: float, output_shape: tuple) -> np.ndarray:
    """
    Args:
        top_points:  all annotated top points where each point is of the shape (x, y).
        radius:  indicates the size of the points.
        output_shape:  contains height and width of desired label map.
    
    Returns:
        label:  labelmap with circles on the positions of the annotated points.
    """
    # create an empty image to add the labels to
    label = np.zeros(output_shape, dtype=np.uint8)
    # set the top point pixel to 1 in the point map
    for x, y in top_points:
        label[round(y), round(x)] = 1
    # dilate the point and add it to the label map
    label = binary_dilation(label, selem=disk(radius))
    label = (label*255).astype(np.uint8)

    return label

# Reference: https://stackoverflow.com/questions/44945111/how-to-efficiently-compute-the-heat-map-of-two-gaussian-distribution-in-python 
# (Not used)
def create_gaussian_label(
    top_points: list, 
    radius: float, 
    output_shape: tuple,
    amplitude: float = 1,
    clip_to_max: bool = True
) -> np.ndarray:
    """
    Args:
        top_points:  all annotated top points where each point is of the shape (x, y).
        radius:  indicates the size of the gaussians.
        output_shape:  contains height and width of desired label map.
        amplitude:  maximum intensity
        clip_to_max:  indicates whether intensities above the maximum 
                      (e.g. when two gaussians are close together) should be clipped.
    Returns:
        label:  labelmap with gaussians on the positions of the annotated points.
    """
    # get gaussians
    gaussians = []
    for x, y in top_points:
        s = np.eye(2)*(radius**2)
        g = multivariate_normal(mean=(x,y), cov=s)
        gaussians.append(g)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    x_positions = np.arange(0, output_shape[1])
    y_positions = np.arange(0, output_shape[0])
    x_grid, y_grid = np.meshgrid(x_positions, y_positions)
    grid = np.stack([x_grid.ravel(), y_grid.ravel()]).T

    # evaluate kernels at grid points
    heatmap = sum(g.pdf(grid) for g in gaussians)

    # normalizing the heatmap values
    label = heatmap.reshape(output_shape)
    max_val = gaussians[0].pdf(top_points[0])
    label /= max_val
    # clip values above 1
    if clip_to_max:
        label = np.clip(label, 0, 1)
    
    # convert to uint
    label = (label*255).astype(np.uint8)

    return label

def add_background(label_array: np.ndarray, axis: int) -> np.ndarray:
    """ 
    Args:
        label_array:  binary label map in range (0-255).
        axis:  in what dimension to add the background map.
    
    Returns:
        label_array:  input after background labelmap is added (as the conventional first map).
    """
    label_array = np.expand_dims(label_array, axis=axis)
    label_array = np.concatenate((255-label_array, label_array), axis=axis)
    
    return label_array