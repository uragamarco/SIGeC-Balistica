"""
Congruent Matching Cells (CMC) Algorithm Implementation
Based on Song et al. (2013-2014) NIST research

This module implements the CMC algorithm for ballistic image comparison,
specifically designed for cartridge case breech face impression analysis.

References:
- Song, J. (2013). Proposed NIST Ballistics Identification System (NBIS) Using 3D Topography Measurements on Correlation Cells. AFTE Journal, 45(2), 184-194.
- Tong, M., Song, J., & Chu, W. (2015). An Improved Algorithm of Congruent Matching Cells (CMC) Method for Firearm Evidence Identifications. Journal of Research of NIST, 120, 102-112.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, NamedTuple
from dataclasses import dataclass
from scipy import ndimage
from scipy.signal import correlate2d
from scipy.stats import pearsonr
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CMCParameters:
    """Configuration parameters for CMC algorithm"""
    # Grid parameters
    num_cells_x: int = 8  # Number of cells in X direction
    num_cells_y: int = 8  # Number of cells in Y direction
    cell_overlap: float = 0.1  # Overlap between adjacent cells (10%)
    
    # Correlation thresholds (based on NIST research)
    ccf_threshold: float = 0.2  # Cross-correlation function threshold (TCCF)
    theta_threshold: float = 15.0  # Angular threshold in degrees (Tθ)
    x_threshold: float = 20.0  # X translation threshold in pixels (Tx)
    y_threshold: float = 20.0  # Y translation threshold in pixels (Ty)
    
    # Quality filters
    min_valid_pixels: float = 0.15  # Minimum 15% valid pixels per cell
    max_missing_ratio: float = 0.85  # Maximum 85% missing pixels allowed
    
    # CMC decision threshold
    cmc_threshold: int = 6  # Minimum CMCs for positive identification (C = 6)
    
    # Advanced parameters
    use_convergence: bool = True  # Use convergence algorithm improvement
    bidirectional: bool = True  # Use forward and backward correlation

class CMCCell(NamedTuple):
    """Represents a single correlation cell"""
    row: int
    col: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    data: np.ndarray
    valid_pixels: int
    missing_ratio: float

class CMCResult(NamedTuple):
    """Result of CMC correlation for a cell pair"""
    cell_index: Tuple[int, int]
    ccf_max: float  # Maximum cross-correlation value
    theta: float    # Rotation angle
    x_offset: float # X translation
    y_offset: float # Y translation
    is_cmc: bool   # Whether this cell pair is a CMC
    confidence: float

@dataclass
class CMCMatchResult:
    """Complete CMC matching result"""
    total_cells: int
    valid_cells: int
    cmc_count: int
    cmc_score: float
    is_match: bool
    confidence: float
    cell_results: List[CMCResult]
    convergence_score: Optional[float] = None

class CMCAlgorithm:
    """
    Implementation of the Congruent Matching Cells algorithm
    for ballistic image comparison
    """
    
    def __init__(self, parameters: Optional[CMCParameters] = None):
        """
        Initialize CMC algorithm with parameters
        
        Args:
            parameters: CMC configuration parameters
        """
        self.params = parameters or CMCParameters()
        self.parameters = self.params  # Alias for backward compatibility
        logger.info(f"CMC Algorithm initialized with {self.params.num_cells_x}x{self.params.num_cells_y} grid")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CMC analysis
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1] range
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply Gaussian filter to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_uint8 = (image * 255).astype(np.uint8)
        image_enhanced = clahe.apply(image_uint8)
        image = image_enhanced.astype(np.float32) / 255.0
        
        return image
    
    def divide_into_cells(self, image: np.ndarray) -> List[CMCCell]:
        """
        Divide image into correlation cells with optional overlap
        
        Args:
            image: Input preprocessed image
            
        Returns:
            List of CMC cells
        """
        height, width = image.shape
        cells = []
        
        # Calculate cell dimensions with overlap
        cell_height = height // self.params.num_cells_y
        cell_width = width // self.params.num_cells_x
        
        overlap_y = int(cell_height * self.params.cell_overlap)
        overlap_x = int(cell_width * self.params.cell_overlap)
        
        for row in range(self.params.num_cells_y):
            for col in range(self.params.num_cells_x):
                # Calculate cell boundaries with overlap
                y_start = max(0, row * cell_height - overlap_y)
                y_end = min(height, (row + 1) * cell_height + overlap_y)
                x_start = max(0, col * cell_width - overlap_x)
                x_end = min(width, (col + 1) * cell_width + overlap_x)
                
                # Extract cell data
                cell_data = image[y_start:y_end, x_start:x_end].copy()
                
                # Calculate quality metrics
                valid_pixels = np.sum(~np.isnan(cell_data) & (cell_data > 0))
                total_pixels = cell_data.size
                missing_ratio = 1.0 - (valid_pixels / total_pixels)
                
                cell = CMCCell(
                    row=row,
                    col=col,
                    x_start=x_start,
                    y_start=y_start,
                    x_end=x_end,
                    y_end=y_end,
                    data=cell_data,
                    valid_pixels=valid_pixels,
                    missing_ratio=missing_ratio
                )
                
                cells.append(cell)
        
        logger.info(f"Created {len(cells)} cells from {width}x{height} image")
        return cells
    
    def calculate_cross_correlation(self, cell1: np.ndarray, cell2: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate cross-correlation between two cells using FFT
        
        Args:
            cell1: Reference cell data
            cell2: Target cell data
            
        Returns:
            Tuple of (max_correlation, x_offset, y_offset)
        """
        # Ensure both cells have the same size
        min_height = min(cell1.shape[0], cell2.shape[0])
        min_width = min(cell1.shape[1], cell2.shape[1])
        
        cell1_crop = cell1[:min_height, :min_width]
        cell2_crop = cell2[:min_height, :min_width]
        
        # Handle NaN values
        mask1 = ~np.isnan(cell1_crop)
        mask2 = ~np.isnan(cell2_crop)
        common_mask = mask1 & mask2
        
        if np.sum(common_mask) < (cell1_crop.size * self.params.min_valid_pixels):
            return 0.0, 0.0, 0.0
        
        # Replace NaN with mean for correlation calculation
        cell1_clean = cell1_crop.copy()
        cell2_clean = cell2_crop.copy()
        cell1_clean[~common_mask] = np.nanmean(cell1_crop)
        cell2_clean[~common_mask] = np.nanmean(cell2_crop)
        
        # Calculate normalized cross-correlation using FFT
        correlation = correlate2d(cell1_clean, cell2_clean, mode='full')
        
        # Find maximum correlation and its position
        max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        max_corr = correlation[max_idx]
        
        # Calculate offsets
        center_y, center_x = np.array(correlation.shape) // 2
        y_offset = max_idx[0] - center_y
        x_offset = max_idx[1] - center_x
        
        # Normalize correlation value
        norm_factor = np.sqrt(np.sum(cell1_clean**2) * np.sum(cell2_clean**2))
        if norm_factor > 0:
            max_corr = max_corr / norm_factor
        
        return float(max_corr), float(x_offset), float(y_offset)
    
    def calculate_rotation_angle(self, cell1: np.ndarray, cell2: np.ndarray, 
                               x_offset: float, y_offset: float) -> float:
        """
        Estimate rotation angle between two cells
        
        Args:
            cell1: Reference cell
            cell2: Target cell
            x_offset: X translation offset
            y_offset: Y translation offset
            
        Returns:
            Rotation angle in degrees
        """
        # Use gradient-based approach for rotation estimation
        try:
            # Calculate gradients
            grad_x1, grad_y1 = np.gradient(cell1)
            grad_x2, grad_y2 = np.gradient(cell2)
            
            # Calculate dominant gradient directions
            angle1 = np.arctan2(np.mean(grad_y1), np.mean(grad_x1))
            angle2 = np.arctan2(np.mean(grad_y2), np.mean(grad_x2))
            
            # Calculate relative rotation
            theta = np.degrees(angle2 - angle1)
            
            # Normalize to [-180, 180] range
            while theta > 180:
                theta -= 360
            while theta < -180:
                theta += 360
                
            return float(theta)
            
        except Exception as e:
            logger.warning(f"Rotation calculation failed: {e}")
            return 0.0
    
    def is_congruent_matching_cell(self, ccf_max: float, theta: float, 
                                 x_offset: float, y_offset: float,
                                 median_theta: float, median_x: float, 
                                 median_y: float) -> bool:
        """
        Determine if a cell pair is a Congruent Matching Cell
        
        Args:
            ccf_max: Maximum cross-correlation value
            theta: Rotation angle
            x_offset: X translation offset
            y_offset: Y translation offset
            median_theta: Median rotation angle (consensus)
            median_x: Median X offset (consensus)
            median_y: Median Y offset (consensus)
            
        Returns:
            True if cell pair is a CMC
        """
        # Check all four CMC criteria
        ccf_ok = ccf_max >= self.params.ccf_threshold
        theta_ok = abs(theta - median_theta) <= self.params.theta_threshold
        x_ok = abs(x_offset - median_x) <= self.params.x_threshold
        y_ok = abs(y_offset - median_y) <= self.params.y_threshold
        
        return ccf_ok and theta_ok and x_ok and y_ok
    
    def calculate_convergence_score(self, cell_results: List[CMCResult]) -> float:
        """
        Calculate convergence score based on CMC distribution
        
        Args:
            cell_results: List of cell correlation results
            
        Returns:
            Convergence score (higher = better convergence)
        """
        if not cell_results:
            return 0.0
        
        # Extract parameters from CMC cells only
        cmc_cells = [r for r in cell_results if r.is_cmc]
        
        if len(cmc_cells) < 3:
            return 0.0
        
        # Calculate standard deviations of parameters
        thetas = [r.theta for r in cmc_cells]
        x_offsets = [r.x_offset for r in cmc_cells]
        y_offsets = [r.y_offset for r in cmc_cells]
        
        theta_std = np.std(thetas)
        x_std = np.std(x_offsets)
        y_std = np.std(y_offsets)
        
        # Convergence score: lower standard deviation = higher convergence
        convergence = 1.0 / (1.0 + theta_std/10.0 + x_std/10.0 + y_std/10.0)
        
        return float(convergence)
    
    def compare_images(self, image1: np.ndarray, image2: np.ndarray) -> CMCMatchResult:
        """
        Compare two ballistic images using CMC algorithm with vectorized operations
        
        Args:
            image1: Reference image
            image2: Target image
            
        Returns:
            CMC matching result
        """
        logger.info("Starting CMC comparison with vectorized operations")
        
        # Preprocess images
        img1_proc = self.preprocess_image(image1)
        img2_proc = self.preprocess_image(image2)
        
        # Divide into cells
        cells1 = self.divide_into_cells(img1_proc)
        cells2 = self.divide_into_cells(img2_proc)
        
        # Filter valid cells
        valid_cells1 = [c for c in cells1 if c.missing_ratio <= self.params.max_missing_ratio]
        valid_cells2 = [c for c in cells2 if c.missing_ratio <= self.params.max_missing_ratio]
        
        logger.info(f"Valid cells: {len(valid_cells1)} from image1, {len(valid_cells2)} from image2")
        
        # Create lookup dictionary for faster cell matching
        cells2_dict = {(c.row, c.col): c for c in valid_cells2}
        
        # Vectorized correlation processing
        cell_results = []
        correlation_params = []
        
        # Batch process correlations for better performance
        valid_pairs = []
        for cell1 in valid_cells1:
            cell_key = (cell1.row, cell1.col)
            if cell_key in cells2_dict:
                valid_pairs.append((cell1, cells2_dict[cell_key]))
        
        if not valid_pairs:
            logger.warning("No valid cell pairs found for comparison")
            return CMCMatchResult(
                total_cells=len(cells1),
                valid_cells=0,
                cmc_count=0,
                cmc_score=0.0,
                is_match=False,
                confidence=0.0,
                cell_results=[]
            )
        
        # Vectorized correlation calculations
        ccf_values = []
        theta_values = []
        x_offsets = []
        y_offsets = []
        
        for cell1, cell2 in valid_pairs:
            # Calculate cross-correlation
            ccf_max, x_offset, y_offset = self.calculate_cross_correlation(
                cell1.data, cell2.data
            )
            
            # Calculate rotation angle
            theta = self.calculate_rotation_angle(
                cell1.data, cell2.data, x_offset, y_offset
            )
            
            # Store parameters for vectorized consensus calculation
            if ccf_max > 0:
                ccf_values.append(ccf_max)
                theta_values.append(theta)
                x_offsets.append(x_offset)
                y_offsets.append(y_offset)
                correlation_params.append((ccf_max, theta, x_offset, y_offset))
        
        # Vectorized consensus calculation using NumPy
        if correlation_params:
            ccf_array = np.array(ccf_values)
            theta_array = np.array(theta_values)
            x_array = np.array(x_offsets)
            y_array = np.array(y_offsets)
            
            # Use NumPy median for faster computation
            median_theta = np.median(theta_array)
            median_x = np.median(x_array)
            median_y = np.median(y_array)
            
            # Vectorized CMC determination
            theta_diff = np.abs(theta_array - median_theta)
            x_diff = np.abs(x_array - median_x)
            y_diff = np.abs(y_array - median_y)
            
            # Vectorized threshold checks
            ccf_valid = ccf_array >= self.params.ccf_threshold
            theta_valid = theta_diff <= self.params.theta_threshold
            x_valid = x_diff <= self.params.x_threshold
            y_valid = y_diff <= self.params.y_threshold
            
            # Combined CMC validity check
            is_cmc_array = ccf_valid & theta_valid & x_valid & y_valid
            cmc_count = np.sum(is_cmc_array)
            
            # Vectorized confidence calculation
            param_consistency = 1.0 - (
                theta_diff / max(self.params.theta_threshold, 1) +
                x_diff / max(self.params.x_threshold, 1) +
                y_diff / max(self.params.y_threshold, 1)
            ) / 3.0
            
            confidence_array = ccf_array * np.maximum(0.0, param_consistency)
            
        else:
            median_theta = median_x = median_y = 0.0
            cmc_count = 0
            is_cmc_array = np.array([])
            confidence_array = np.array([])
        
        # Create detailed results
        for i, ((cell1, cell2), (ccf_max, theta, x_offset, y_offset)) in enumerate(zip(valid_pairs, correlation_params)):
            cell_idx = (cell1.row, cell1.col)
            is_cmc = bool(is_cmc_array[i]) if i < len(is_cmc_array) else False
            confidence = float(confidence_array[i]) if i < len(confidence_array) else 0.0
            
            result = CMCResult(
                cell_index=cell_idx,
                ccf_max=ccf_max,
                theta=theta,
                x_offset=x_offset,
                y_offset=y_offset,
                is_cmc=is_cmc,
                confidence=confidence
            )
            
            cell_results.append(result)
        
        # Calculate overall results
        total_cells = len(cells1)
        valid_cells = len(valid_cells1)
        is_match = bool(cmc_count >= self.params.cmc_threshold)
        
        # Calculate CMC score (normalized)
        cmc_score = cmc_count / max(valid_cells, 1) if valid_cells > 0 else 0.0
        
        # Calculate overall confidence
        if cell_results:
            avg_confidence = np.mean([r.confidence for r in cell_results])
            match_confidence = avg_confidence * (cmc_count / max(self.params.cmc_threshold, 1))
        else:
            match_confidence = 0.0
        
        # Calculate convergence score if enabled
        convergence_score = None
        if self.params.use_convergence:
            convergence_score = self.calculate_convergence_score(cell_results)
            # Adjust match confidence with convergence
            if convergence_score > 0:
                match_confidence *= (1.0 + convergence_score)
        
        result = CMCMatchResult(
            total_cells=total_cells,
            valid_cells=valid_cells,
            cmc_count=cmc_count,
            cmc_score=cmc_score,
            is_match=is_match,
            confidence=min(1.0, match_confidence),
            cell_results=cell_results,
            convergence_score=convergence_score
        )
        
        logger.info(f"CMC Analysis complete: {cmc_count} CMCs found, Match: {is_match}")
        return result
    
    def bidirectional_comparison(self, image1: np.ndarray, image2: np.ndarray) -> CMCMatchResult:
        """
        Perform bidirectional CMC comparison (A vs B and B vs A)
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Combined CMC result
        """
        if not self.params.bidirectional:
            return self.compare_images(image1, image2)
        
        logger.info("Performing bidirectional CMC comparison")
        
        # Forward comparison (A vs B)
        result_forward = self.compare_images(image1, image2)
        
        # Backward comparison (B vs A)
        result_backward = self.compare_images(image2, image1)
        
        # Combine results (take the better result or average)
        combined_cmc_count = max(result_forward.cmc_count, result_backward.cmc_count)
        combined_confidence = (result_forward.confidence + result_backward.confidence) / 2.0
        combined_score = (result_forward.cmc_score + result_backward.cmc_score) / 2.0
        
        # Use convergence from the better result
        convergence_score = None
        if result_forward.convergence_score and result_backward.convergence_score:
            convergence_score = max(result_forward.convergence_score, result_backward.convergence_score)
        
        is_match = combined_cmc_count >= self.params.cmc_threshold
        
        # Combine cell results (use forward as primary)
        combined_result = CMCMatchResult(
            total_cells=result_forward.total_cells,
            valid_cells=result_forward.valid_cells,
            cmc_count=combined_cmc_count,
            cmc_score=combined_score,
            is_match=is_match,
            confidence=combined_confidence,
            cell_results=result_forward.cell_results,
            convergence_score=convergence_score
        )
        
        logger.info(f"Bidirectional CMC complete: Forward={result_forward.cmc_count}, "
                   f"Backward={result_backward.cmc_count}, Combined={combined_cmc_count}")
        
        return combined_result
    
    def compare_images_bidirectional(self, image1: np.ndarray, image2: np.ndarray) -> CMCMatchResult:
        """
        Alias for bidirectional_comparison for backward compatibility
        """
        return self.bidirectional_comparison(image1, image2)

def create_cmc_algorithm(config: Optional[Dict] = None) -> CMCAlgorithm:
    """
    Factory function to create CMC algorithm instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured CMC algorithm instance
    """
    if config:
        params = CMCParameters(**config)
    else:
        params = CMCParameters()
    
    return CMCAlgorithm(parameters=params)

# Example usage and testing
if __name__ == "__main__":
    # Create test images
    test_img1 = np.random.rand(400, 400).astype(np.float32)
    test_img2 = test_img1 + 0.1 * np.random.rand(400, 400).astype(np.float32)
    
    # Create CMC algorithm
    cmc = create_cmc_algorithm()
    
    # Perform comparison
    result = cmc.bidirectional_comparison(test_img1, test_img2)
    
    print(f"CMC Analysis Results:")
    print(f"Total Cells: {result.total_cells}")
    print(f"Valid Cells: {result.valid_cells}")
    print(f"CMC Count: {result.cmc_count}")
    print(f"CMC Score: {result.cmc_score:.3f}")
    print(f"Is Match: {result.is_match}")
    print(f"Confidence: {result.confidence:.3f}")
    if result.convergence_score:
        print(f"Convergence Score: {result.convergence_score:.3f}")