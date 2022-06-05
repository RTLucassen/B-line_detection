"""
Utility functions for transformation of images or points from polar to Cartesian coordinates,
as well as transformation of arrays from Cartesian to polar.
"""

from typing import Union

import numpy as np
from skimage.transform import resize
from polarTransform import convertToPolarImage, convertToCartesianImage

# ----------------------------  RADII & ANGLES  ----------------------------

# source: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

def line(p1: tuple, p2: tuple) -> tuple:
    """
    Args:
        p1:  first point (x, y)
        p2:  second point (x, y)
    
    Returns:
        line:  three values that parameterizes the line through both points.
    """
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    
    line = (A, B, -C)
    
    return line

def find_intersection(L1: tuple, L2: tuple) -> Union[tuple, bool]:
    """ 
    Args:
        L1:  first line, parameterized by three values.
        L2:  second line, parameterized by three values.

    Returns:
        output:  if valid, the intersection point coordinate, else False.
    """
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    
    if D != 0:
        x = Dx / D
        y = Dy / D
        output = (x, y)
    else:
        output = False
   
    return output

def euclidean_dist(p1: tuple, p2: tuple) -> float:
    """
    Args:
        p1:  first point
        p2:  second point
    
    Returns:
        euclidean distance
    """
    if len(p1) != len(p2):
        raise ValueError('p1 and p2 must be equal in length')
    return np.sqrt(sum([(p2[i]-p1[i])**2 for i in np.arange(len(p1))]))

def angle(L1: tuple, L2: tuple) -> float:
    """ 
    Args:
        L1: first line, parameterized by three values.
        L2: second line, parameterized by three values.

    Returns:
        angle between two lines in degrees.
    """
    slope_L1 = L1[0]/L1[1]
    slope_L2 = L2[0]/L2[1]

    angle_L1 = np.arctan(slope_L1)
    angle_L2 = np.arctan(slope_L2)

    return (np.pi - angle_L1 + angle_L2)*180/np.pi

def extract_radii_angles(points: tuple) -> tuple:
    """    
    Args:
        points:  four annotated corner points
    
    Returns:
        intersection:  coordinates of intersection point of the two lines spanned by both point pairs.
        min_radius:  shortest distance between intersection point and any of the two top points.
        max_radius:  largest distance between intersection point and any of the two bottom points.
        min_angle:  angle between left line and reference (on left side).
        max_angle:  angle between right line and reference (on right side).
    """
    top_left, bottom_left, top_right, bottom_right = points

    # create lines using the points
    L1 = line(top_left, bottom_left)
    L2 = line(top_right, bottom_right)

    # get the intersection point
    intersection = find_intersection(L1, L2)
    intersection = (intersection[0], intersection[1])

    # get the minimum and maximum radius for both the left and right side
    top_left_radius = euclidean_dist(intersection, top_left)
    bottom_left_radius = euclidean_dist(intersection, bottom_left)
    top_right_radius = euclidean_dist(intersection, top_right)
    bottom_right_radius = euclidean_dist(intersection, bottom_right)

    # get the minimum and maximum radius
    min_radius = min(top_left_radius, top_right_radius)  
    max_radius = max(bottom_left_radius, bottom_right_radius)

    # get the angle between the two lines of the fan-shaped image data
    total_angle = angle(L1, L2)

    # get the minimum and maximum angle
    min_angle = (90-total_angle/2) 
    max_angle = (90+total_angle/2)

    return intersection, min_radius, max_radius, min_angle, max_angle

# -----------------------------  PROCESSING  -----------------------------

def crop_and_resize(
    image: np.ndarray, 
    points: tuple, 
    output_shape: tuple, 
    order: int = 1, 
    apply_mask: bool = True
) -> tuple:
    """ 
    Crop the image based on the four annotated corner points indicating the region of interest, 
    add padding to get the correct aspect ratio, and resize the image to the given output shape.

    Args:
        image:  input image
        points:  four annotated corner points.
        output_shape:  shape of output image.
        order:  interpolation order for resizing image.
        apply_mask:  indicates if the boolean mask should be applied to remove all embedded information from the sides.
    
    Returns:
        processed_image: input image after cropping, padding, and resizing.
        processing information
    """  
    # construct a mask from the points to remove embedded writing and to find the region of interest        
    mask = cartesian2polar_image(np.ones(output_shape), points, image.shape)

    # get the maximum intensity for every row and every column
    mip_horizontal = np.max(mask, axis=0)
    mip_vertical   = np.max(mask, axis=1)
    # find the corner points of the rectangular region of interest
    top    = len(mip_vertical)-len(np.trim_zeros(mip_vertical, trim='f'))
    bottom = len(np.trim_zeros(mip_vertical, trim='b'))
    left   = len(mip_horizontal)-len(np.trim_zeros(mip_horizontal, trim='f'))
    right  = len(np.trim_zeros(mip_horizontal, trim='b'))
    
    # crop the unprocessed image to only contain the region of interest
    if apply_mask:
        roi = (image*mask)[top:bottom, left:right]
    else:
        roi = (image*np.ones_like(mask))[top:bottom, left:right]
    roi_shape = roi.shape

    # find the amount of padding needed to get the correct aspect ratio
    if (roi_shape[0]/output_shape[0])*output_shape[1]-roi_shape[1] > 0:
        difference = np.ceil((roi_shape[0]/output_shape[0])*output_shape[1]-roi_shape[1])
        if difference % 2 == 1:
            add_left, add_right = int(difference//2), int(difference//2+1)
        else:
            add_left = add_right = int(difference//2)
        padding = ((0, 0), (add_left, add_right))
    
    else:
        difference = np.ceil((roi_shape[1]/output_shape[1])*output_shape[0]-roi_shape[0])
        if difference % 2 == 1:
            add_top, add_bottom =  int(difference//2), int(difference//2+1)
        else:
            add_top = add_bottom = int(difference//2)
        padding = ((add_top, add_bottom), (0, 0))
    
    # add padding to the region of interest to create the same aspect ratio
    roi = np.pad(roi, pad_width=padding, mode='constant')
    
    # calculate the scaling factor for resizing
    scaling_factor = output_shape[0]/roi.shape[0]
    # resize the image   
    processed_image = resize(roi, output_shape, anti_aliasing=False, order=order).astype(np.uint8)

    return processed_image, (((top,bottom), (left,right)), padding, scaling_factor)

def process_point(point: tuple, processing_info: tuple) -> tuple:
    """
    Args:
        point:  coordinate (x, y)
        processing_info:  information about cropping, padding and scaling of images.

    Returns:
        point:  original point (x, y) after processing.
    """
    cropping, padding, scaling = processing_info
    
    # account for cropping
    point = (point[0]-cropping[1][0], point[1]-cropping[0][0])
    # account for padding
    point = (point[0]+padding[1][0], point[1]+padding[0][0])
    # account for resizing
    point = (point[0]*scaling, point[1]*scaling)
    
    return point

def polar2cartesian_image(image: np.ndarray, points: list, output_shape: tuple, order: int = 3) -> np.ndarray:
    """ 
    Args:
        image:  input image for transformation (height, width)
        points:  four annotated corner points that correspond to the image.
        output_shape:  desired shape of output image
        order:  interpolation order used for transformation from polar to Cartesian.
    
    Returns:
        cartesian_image:  selected part of input image after transformation from polar to Cartesian.
    """
    # check the input arguments
    if len(image.shape) != 2:
        raise ValueError('Image has an invalid number of dimensions.')
    if len(output_shape) != 2:
        raise ValueError('Output shape has an invalid number of dimensions.')

    # get the intersection, radii, and angles based on the four corner points
    intersection, min_radius, max_radius, min_angle, max_angle = extract_radii_angles(points)

    # create the cartesian array
    cartesian_image = convertToPolarImage(
        image, 
        center = intersection, 
        initialAngle = min_angle/180*np.pi, 
        finalAngle = max_angle/180*np.pi, 
        initialRadius = min_radius, 
        finalRadius = max_radius, 
        radiusSize = output_shape[0], 
        angleSize = output_shape[1],
        order = order)

    cartesian_image = np.fliplr(cartesian_image[0].T)

    return cartesian_image

def cartesian2polar_image(image: np.ndarray, points: list, output_shape: tuple, order: int = 3) -> np.ndarray:
    """ 
    Args:
        image:  input image for transformation of the shape: (height, width).
        points:  four annotated corner points that correspond to the image.
        output_shape:  desired shape of output image.
        order:  interpolation order used for transformation from polar to Cartesian.
    
    Returns:
        cartesian_image:  selected part of input image after transformation from polar to Cartesian.
    """
    # get the intersection, radii, and angles based on the four corner points
    intersection, min_radius, max_radius, min_angle, max_angle = extract_radii_angles(points)

    # change the axis order and mirror the image
    image = np.swapaxes(image[None, ...], 1,2)
    image = image[:, ::-1, :]

    # create a cartesian array
    cartesian_image = convertToCartesianImage(
        image, 
        center = intersection, 
        initialAngle = min_angle/180*np.pi, 
        finalAngle = max_angle/180*np.pi, 
        initialRadius = min_radius, 
        finalRadius = max_radius,
        imageSize = output_shape,
        order = order)

    return cartesian_image[0][0, ...]