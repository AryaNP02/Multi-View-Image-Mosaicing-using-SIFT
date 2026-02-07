# Multi-View Image Mosaicing using SIFT

A Python implementation for creating panoramic images by stitching multiple overlapping images using SIFT features and homography estimation.

## Overview

This project implements an automated image stitching pipeline that combines multiple images with overlapping regions into a single seamless panorama. The implementation uses computer vision techniques including SIFT feature detection, feature matching, RANSAC-based homography estimation, and perspective warping.

### Key Features

- **SIFT-based Feature Detection**: Robust keypoint detection and descriptor extraction
- **Intelligent Feature Matching**: KNN matcher with Lowe's ratio test for filtering spurious matches
- **RANSAC Homography Estimation**: Robust estimation of geometric transformations between images
- **Multiple Stitching Modes**:
  - **Ordered Mode**: Sequential stitching for images in left-to-right order
  - **Unordered Mode**: Automatic detection of best matching pairs for arbitrary image ordering
- **Visualization Tools**: Match visualization and stitching process inspection
- **Configurable Parameters**: YAML-based configuration for easy parameter tuning

## Technology Stack

### Core Libraries

- **OpenCV (cv2)**: Computer vision operations including SIFT, feature matching, and image warping
- **NumPy**: Numerical computations and matrix operations
- **Matplotlib**: Visualization of results and intermediate steps
- **PyYAML**: Configuration file parsing

### Algorithms Used

1. **SIFT (Scale-Invariant Feature Transform)**: Feature detection and description
2. **KNN (K-Nearest Neighbors)**: Feature matching with k=2
3. **RANSAC (Random Sample Consensus)**: Robust homography estimation
4. **SVD (Singular Value Decomposition)**: Solving homography linear equations

## Methodology

### 1. Feature Detection

The pipeline begins by detecting SIFT features in each input image:

```python
def detect_features(img: np.ndarray) -> Tuple[list, np.ndarray]
```

- Identifies distinctive keypoints that are invariant to scale and rotation
- Computes 128-dimensional descriptors for each keypoint
- Uses OpenCV's SIFT implementation (`cv2.xfeatures2d.SIFT_create()`)

### 2. Feature Matching

Matches are found between pairs of images using KNN matching:

```python
def match_images(img1: np.ndarray, img2: np.ndarray)
```

**Process:**
- Detects features in both images
- Uses Brute-Force matcher with KNN (k=2)
- Applies **Lowe's ratio test** (threshold: 0.75) to filter unreliable matches
- Only retains matches where the distance to the nearest neighbor is significantly less than the distance to the second-nearest neighbor

### 3. Homography Estimation

Computes the geometric transformation between image pairs using RANSAC:

```python
def find_homography_matrix(pts1: np.ndarray, pts2: np.ndarray, 
                          iterations: int = 100, threshold: int = 15)
```

**RANSAC Algorithm:**
1. Randomly sample 4 point correspondences
2. Construct the homography matrix equation system:
   - For each point pair (x₁, y₁) ↔ (x₂, y₂), create two equations
   - Build an 8×9 matrix M from the 4 point pairs
3. Solve using SVD to find the homography matrix H
4. Compute reprojection error for all points
5. Count inliers (points with error < threshold)
6. Repeat for specified iterations and keep the best H

**Homography Matrix:**
The 3×3 transformation matrix that maps points from one image plane to another:
```
x₁ = H × x₂
```

### 4. Image Warping

Transforms the second image to align with the first image's plane:

```python
def stitch_two_images(img1: np.ndarray, img2: np.ndarray, H: np.ndarray)
```

**Process:**
- Uses `cv2.warpPerspective()` to apply the homography transformation
- Handles proper canvas sizing to accommodate the warped image
- Manages overlapping regions by creating binary masks

### 5. Image Blending

Combines the warped image with the base image:

**Blending Strategy:**
- Identifies overlapping regions using binary masks
- Non-overlapping regions from both images are preserved
- Overlapping regions use pixels from the base image (simple blending)
- Trims zero-valued rows and columns from the result

### 6. Multi-Image Stitching

#### Ordered Stitching
For images captured in sequence (left-to-right):

```python
def stitch_multiple_ordered(paths: List[str])
```

**Algorithm:**
1. Start with the first image as the base
2. For each subsequent image:
   - Find matches with current panorama
   - Estimate homography
   - Warp and blend the new image
   - Update the panorama
3. Continue until all images are processed

#### Unordered Stitching
For images in arbitrary order:

```python
def stitch_multiple_unordered(paths: List[str])
```

**Algorithm:**
1. **Base Selection**: Find image with maximum total matches across all images
2. **Iterative Stitching**:
   - Find the image with most matches to current base
   - Stitch them together
   - Replace both images with the stitched result
   - Repeat until only one image remains
3. **Padding Strategy**: Images are padded to allow flexible positioning

**Advantages:**
- No need for manual ordering
- More robust to varying overlap patterns
- Better handles complex panorama structures

## Usage

### Basic Command Line Usage

```bash
# Stitch images in ordered mode
python main.py --images img1.jpg img2.jpg img3.jpg --mode ordered --output panorama.jpg

# Stitch images in unordered mode
python main.py --images img1.jpg img2.jpg img3.jpg --mode unordered --output panorama.jpg

# Visualize matches between two images
python main.py --visualize-matches img1.jpg img2.jpg

# Visualize complete stitching process
python main.py --visualize-stitch img1.jpg img2.jpg --output ./visualizations
```

### Configuration File

Edit `config.yaml` to customize parameters:

```yaml
# Matching Parameters
matching:
  knn_match_k: 2
  distance_ratio: 0.75  # Lowe's ratio test threshold

# RANSAC Parameters
ransac:
  iterations: 100       # Number of RANSAC iterations
  error_threshold: 15   # Pixel error threshold for inliers
  min_inliers: 4

# Stitching Mode
stitching:
  mode: "ordered"       # "ordered" or "unordered"
  visualization: true   # Show results
```



## Algorithm Parameters

### SIFT Features
- Automatically detects keypoints at multiple scales
- 128-dimensional descriptors per keypoint

### Feature Matching
- **KNN k-value**: 2 (finds two nearest neighbors)
- **Distance ratio**: 0.75 (Lowe's ratio test threshold)

### RANSAC
- **Iterations**: 100 (configurable)
- **Error threshold**: 15 pixels (configurable)
- **Minimum points**: 4 (required for homography)



## References

- Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
- Fischler, M. A., & Bolles, R. C. (1981). "Random Sample Consensus: A Paradigm for Model Fitting"
- Szeliski, R. (2006). "Image Alignment and Stitching: A Tutorial"

