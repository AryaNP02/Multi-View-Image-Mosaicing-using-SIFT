"""
Image Stitching and Panorama Generation Package
A Python library for stitching multiple images into panoramas using SIFT features.
"""

__version__ = '1.0.0'
__author__ = 'Image Stitching Team'

from core.utils import (
    detect_features,
    match_images,
    get_matching_coordinates,
    find_homography_matrix,
    trim_image,
    stitch_two_images,
    stitch_multiple_ordered,
    stitch_multiple_unordered
)

__all__ = [
    'detect_features',
    'match_images',
    'get_matching_coordinates',
    'find_homography_matrix',
    'trim_image',
    'stitch_two_images',
    'stitch_multiple_ordered',
    'stitch_multiple_unordered'
]
