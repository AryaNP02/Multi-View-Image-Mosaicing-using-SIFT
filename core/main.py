"""
Main Image Stitching and Panorama Generation Script
Entry point for the image stitching application.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from utils import (
    match_images,
    get_matching_coordinates,
    find_homography_matrix,
    stitch_two_images,
    stitch_multiple_ordered,
    stitch_multiple_unordered,
    detect_features
)


class StitchingConfig:
    """Configuration handler for image stitching."""
    
    def __init__(self, config_file: str = None):
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to config.yaml file
        """
        if config_file is None:
            config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_file}")
            self.config = self._get_default_config()
    
    @staticmethod
    def _get_default_config():
        """Return default configuration."""
        return {
            'matching': {'distance_ratio': 0.75},
            'ransac': {'iterations': 100, 'error_threshold': 15},
            'stitching': {'mode': 'ordered', 'visualization': True},
            'paths': {'output_dir': './output'}
        }
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)


def setup_logger(log_level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('ImageStitching')
    logger.setLevel(getattr(logging, log_level))
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def stitch_images(image_paths: List[str], config: StitchingConfig, 
                  logger: logging.Logger, output_path: Optional[str] = None) -> np.ndarray:
    """
    Stitch multiple images into a panorama.
    
    Args:
        image_paths: List of image file paths
        config: Configuration object
        logger: Logger instance
        output_path: Optional path to save result
        
    Returns:
        Stitched panorama image
    """
    logger.info(f"Starting image stitching with {len(image_paths)} images")
    logger.info(f"Stitching mode: {config.get('stitching', {}).get('mode', 'ordered')}")
    
    try:
        mode = config.get('stitching', {}).get('mode', 'ordered')
        
        if mode == 'ordered':
            logger.info("Using ordered stitching (sequential matching)")
            result = stitch_multiple_ordered(image_paths)
        else:
            logger.info("Using unordered stitching (best-match based)")
            result = stitch_multiple_unordered(image_paths)
        
        logger.info("Stitching completed successfully")
        
        # Save result if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            cv2.imwrite(output_path, result)
            logger.info(f"Result saved to {output_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error during stitching: {str(e)}")
        raise


def visualize_matches(img1_path: str, img2_path: str, output_path: Optional[str] = None) -> None:
    """
    Visualize matches between two images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_path: Optional path to save visualization
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error reading images")
        return
    
    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    p1, d1, p2, d2, matching_inds, top = match_images(img1, img2)
    
    # Draw matches
    matched_img = cv2.drawMatchesKnn(img1, p1, img2, p2, top, None, flags=2)
    matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(matched_img_rgb)
    plt.title(f"Matched Features ({len(top)} matches)")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_stitching(img1_path: str, img2_path: str, output_dir: Optional[str] = None) -> None:
    """
    Visualize the complete stitching process for two images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_dir: Optional directory to save visualizations
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error reading images")
        return
    
    # Find matches
    p1, d1, p2, d2, matching_inds, top = match_images(img1, img2)
    
    if matching_inds.shape[0] < 4:
        print("Not enough matches found")
        return
    
    # Get coordinates
    pts1, pts2 = get_matching_coordinates(p1, p2, matching_inds)
    
    # Find homography
    H = find_homography_matrix(pts1, pts2)
    
    # Perform stitching
    result = stitch_two_images(img1, img2, H)
    
    # Convert to RGB for display
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Visualize
    plt.figure(figsize=(20, 10))
    plt.imshow(result_rgb)
    plt.title("Stitched Result")
    plt.axis('off')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'stitched_result.png'), dpi=150, bbox_inches='tight')
    
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Image Stitching and Panorama Generation Tool'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--images',
        nargs='+',
        help='List of image paths to stitch'
    )
    parser.add_argument(
        '--mode',
        choices=['ordered', 'unordered'],
        default='ordered',
        help='Stitching mode'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for stitched image'
    )
    parser.add_argument(
        '--visualize-matches',
        nargs=2,
        metavar=('IMG1', 'IMG2'),
        help='Visualize matches between two images'
    )
    parser.add_argument(
        '--visualize-stitch',
        nargs=2,
        metavar=('IMG1', 'IMG2'),
        help='Visualize stitching process'
    )
    
    args = parser.parse_args()
    
    # Setup
    config = StitchingConfig(args.config)
    logger = setup_logger()
    
    # Handle visualization modes
    if args.visualize_matches:
        logger.info("Visualizing matches...")
        visualize_matches(args.visualize_matches[0], args.visualize_matches[1])
        return
    
    if args.visualize_stitch:
        logger.info("Visualizing stitching process...")
        visualize_stitching(args.visualize_stitch[0], args.visualize_stitch[1], args.output)
        return
    
    # Handle stitching
    if args.images:
        try:
            result = stitch_images(args.images, config, logger, args.output)
            
            # Visualize if requested
            if config.get('stitching', {}).get('visualization', True):
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(20, 10))
                plt.imshow(result_rgb)
                plt.title('Final Stitched Panorama')
                plt.axis('off')
                plt.show()
        
        except Exception as e:
            logger.error(f"Failed to stitch images: {str(e)}")
            return 1
    
    else:
        # Run example if no arguments
        logger.info("No images provided. Running example with config settings...")
        
        image_sets = config.get('paths', {}).get('image_sets', {})
        if image_sets:
            set1_paths = image_sets.get('set1', [])
            if set1_paths:
                try:
                    result = stitch_images(set1_paths, config, logger, 
                                          os.path.join(config.get('paths', {}).get('output_dir', './output'), 
                                                       'example_output.jpg'))
                    
                    if config.get('stitching', {}).get('visualization', True):
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        plt.figure(figsize=(20, 10))
                        plt.imshow(result_rgb)
                        plt.title('Example Panorama')
                        plt.axis('off')
                        plt.show()
                
                except Exception as e:
                    logger.error(f"Example execution failed: {str(e)}")
        else:
            logger.warning("No image sets found in configuration")
    
    return 0


if __name__ == '__main__':
    exit(main())
