"""
Image Stitching Utilities Module
Provides functions for SIFT feature detection, matching, homography estimation, and image stitching.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def detect_features(img: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Detect SIFT features in an image.
    
    Args:
        img: Input image
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    sift = cv2.xfeatures2d.SIFT_create()
    points, descriptors = sift.detectAndCompute(img, None)
    return points, descriptors


def match_images(img1: np.ndarray, img2: np.ndarray) -> Tuple[list, np.ndarray, list, np.ndarray, np.ndarray, list]:
    """
    Match SIFT features between two images using KNN matcher.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        Tuple of (keypoints1, descriptors1, keypoints2, descriptors2, matching_indices, top_matches)
    """
    # Get features for each image
    p1, d1 = detect_features(img1)
    p2, d2 = detect_features(img2)
    
    if d1 is None or d2 is None:
        return p1, d1, p2, d2, np.array([]), []
    
    # Find nearest neighbours for each image using KNN
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(d1, d2, k=2)
    
    # Find good matches using Lowe's ratio test
    top = []
    matching_inds = []
    for match_pair in matches:
        if len(match_pair) == 2:
            a, b = match_pair
            if a.distance < 0.75 * b.distance:
                top.append([a])
                matching_inds.append((a.queryIdx, a.trainIdx))
    
    return p1, d1, p2, d2, np.array(matching_inds), top


def get_matching_coordinates(p1: list, p2: list, matching_inds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract coordinates of matching points.
    
    Args:
        p1: Keypoints from first image
        p2: Keypoints from second image
        matching_inds: Indices of matches
        
    Returns:
        Tuple of (points1, points2) as numpy arrays
    """
    pts1, pts2 = [], []
    for i in range(matching_inds.shape[0]):
        pts1.append(p1[matching_inds[i, 0]].pt)
        pts2.append(p2[matching_inds[i, 1]].pt)
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    return pts1, pts2


def find_homography_matrix(pts1: np.ndarray, pts2: np.ndarray, 
                          iterations: int = 100, threshold: int = 15) -> np.ndarray:
    """
    Find the homography matrix using RANSAC algorithm.
    
    Args:
        pts1: Points in first image
        pts2: Points in second image
        iterations: Number of RANSAC iterations
        threshold: Error threshold for inliers
        
    Returns:
        Best homography matrix found
    """
    max_correct = 0
    best_H = np.eye(3)
    
    # Run RANSAC for specified iterations
    for i in range(iterations):
        # Select random 4 point correspondences
        inds = np.random.choice(pts1.shape[0], 4, replace=False)
        
        # Build the matrix for solving homography
        M = []
        for j in inds:
            x1, y1 = pts1[j][0], pts1[j][1]
            x2, y2 = pts2[j][0], pts2[j][1]
            M.append([-x2, -y2, -1, 0, 0, 0, x1*x2, x1*y2, x1])
            M.append([0, 0, 0, -x2, -y2, -1, y1*x2, y1*y2, y1])
        
        M = np.array(M, dtype=np.float32)
        
        # Solve using SVD
        try:
            u, s, v = np.linalg.svd(M)
            h = v[-1, :]
            H = np.reshape(h, (3, 3))
        except:
            continue
        
        # Calculate reprojection error
        pts2_aug = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
        proj = np.matmul(H, pts2_aug.T)
        proj = (proj / proj[2, :])[:2, :]
        
        err = np.sqrt(np.sum(np.power(pts1 - proj.T, 2), axis=1))
        err[err > threshold] = 0
        err[err != 0] = 1
        correct = np.sum(err)
        
        if correct > max_correct:
            max_correct = correct
            best_H = H
    
    return best_H


def trim_image(img: np.ndarray) -> np.ndarray:
    """
    Remove rows and columns with all zeros from an image.
    
    Args:
        img: Input image
        
    Returns:
        Trimmed image
    """
    # Remove rows with all zeros
    img = img[~np.all(img == 0, axis=(1, 2))]
    # Remove columns with all zeros
    img = img[:, ~np.all(img == 0, axis=(0, 2))]
    
    return img


def stitch_two_images(img1: np.ndarray, img2: np.ndarray, H: np.ndarray, 
                      order: bool = False) -> np.ndarray:
    """
    Stitch two images using homography matrix.
    
    Args:
        img1: Base image
        img2: Image to be warped and stitched
        H: Homography matrix
        order: If True, allow larger output canvas
        
    Returns:
        Stitched image
    """
    # Warp img2 to img1's plane
    if order:
        output_size = (img1.shape[1] + img2.shape[1], img1.shape[0] + img2.shape[0])
    else:
        output_size = (img1.shape[1], img1.shape[0])
    
    res = cv2.warpPerspective(img2, H, output_size)
    
    # Create masks to find non-overlapping regions
    img1_mask = np.any(img1 != 0, axis=2)
    res_sub = res[0:img1.shape[0], 0:img1.shape[1], :]
    res_mask = np.any(res_sub != 0, axis=2)
    mask = np.logical_not(img1_mask & res_mask)
    
    # Combine the images
    res[0:img1.shape[0], 0:img1.shape[1], :] += img1 * mask[:, :, np.newaxis]
    
    # Remove zero rows and columns
    res = trim_image(res)
    
    return res


def stitch_multiple_ordered(paths: List[str]) -> np.ndarray:
    """
    Stitch multiple images in order (adjacent images have overlapping regions).
    
    Args:
        paths: List of image file paths
        
    Returns:
        Final stitched panorama
    """
    if not paths:
        raise ValueError("Image paths list is empty")
    
    img1 = cv2.imread(paths[0])
    if img1 is None:
        raise FileNotFoundError(f"Could not read image: {paths[0]}")
    
    for p in paths[1:]:
        img2 = cv2.imread(p)
        if img2 is None:
            print(f"Warning: Could not read image {p}, skipping...")
            continue
        
        # Find matches between images
        p1, d1, p2, d2, matching_inds, _ = match_images(img1, img2)
        
        if matching_inds.shape[0] < 4:
            print(f"Warning: Not enough matches found for {p}, skipping...")
            continue
        
        # Get matching coordinates
        pts1, pts2 = get_matching_coordinates(p1, p2, matching_inds)
        
        # Find homography matrix
        H = find_homography_matrix(pts1, pts2)
        
        # Stitch the images
        img1 = stitch_two_images(img1, img2, H, order=True)
    
    return img1


def stitch_multiple_unordered(paths: List[str]) -> np.ndarray:
    """
    Stitch multiple images in unordered manner (finds best matches automatically).
    
    Args:
        paths: List of image file paths
        
    Returns:
        Final stitched panorama
    """
    if not paths:
        raise ValueError("Image paths list is empty")
    
    # Read all images and pad them
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: Could not read image {p}, skipping...")
            continue
        # Pad images for unordered stitching
        img = np.pad(img, ((img.shape[0], img.shape[0]), 
                          (img.shape[1], img.shape[1]), 
                          (0, 0)), 'constant')
        imgs.append(img)
    
    if len(imgs) < 2:
        raise ValueError("Need at least 2 valid images for stitching")
    
    # Find base image (with most matches)
    max_matches = 0
    base_idx = 0
    
    for i in range(len(imgs)):
        total_matches = 0
        for j in range(len(imgs)):
            if i == j:
                continue
            _, _, _, _, matching_inds, _ = match_images(imgs[i], imgs[j])
            total_matches += matching_inds.shape[0]
        
        if total_matches > max_matches:
            max_matches = total_matches
            base_idx = i
    
    # Iteratively stitch images
    while len(imgs) > 1:
        max_matches = 0
        best_other_idx = -1
        best_p1 = best_p2 = best_matching_inds = None
        
        # Find best match for current base image
        for i in range(len(imgs)):
            if i == base_idx:
                continue
            
            p1, d1, p2, d2, matching_inds, _ = match_images(imgs[base_idx], imgs[i])
            
            if matching_inds.shape[0] > max_matches:
                max_matches = matching_inds.shape[0]
                best_other_idx = i
                best_p1, best_p2 = p1, p2
                best_matching_inds = matching_inds
        
        if best_other_idx == -1:
            print("Warning: No good matches found, stopping stitching")
            break
        
        # Extract matching points
        matching_inds = np.array(best_matching_inds)
        pts1, pts2 = get_matching_coordinates(best_p1, best_p2, matching_inds)
        
        # Find homography and stitch
        H = find_homography_matrix(pts1, pts2)
        img_stitched = stitch_two_images(imgs[base_idx], imgs[best_other_idx], H)
        
        # Remove processed images and add stitched result
        remove_indices = sorted([base_idx, best_other_idx], reverse=True)
        for idx in remove_indices:
            del imgs[idx]
        
        # Pad and add stitched image
        img_stitched = np.pad(img_stitched, 
                              ((img_stitched.shape[0], img_stitched.shape[0]), 
                               (img_stitched.shape[1], img_stitched.shape[1]), 
                               (0, 0)), 'constant')
        imgs.append(img_stitched)
        base_idx = len(imgs) - 1
    
    return trim_image(imgs[0]) if imgs else None
