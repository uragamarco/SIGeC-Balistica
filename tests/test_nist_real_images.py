#!/usr/bin/env python3
"""
Test script for validating spatial calibration with real NIST images.
Uses actual metadata from StudyInfo.xlsx to test DPI calibration accuracy.
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image, ExifTags
import piexif
import tempfile
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_processing.spatial_calibration import SpatialCalibrator, CalibrationData
from image_processing.nist_compliance_validator import NISTComplianceValidator
from image_processing.unified_preprocessor import UnifiedPreprocessor
from config.unified_config import ImageProcessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NISTRealImageTester:
    """Test spatial calibration with real NIST images and metadata."""
    
    def __init__(self):
        self.excel_path = 'get_project_root()/uploads/Muestras NIST FADB/Cary Persistence/StudyInfo.xlsx'
        self.images_dir = 'get_project_root()/uploads/Muestras NIST FADB/Cary Persistence/cc'
        self.calibrator = SpatialCalibrator()
        self.validator = NISTComplianceValidator()
        self.preprocessor = UnifiedPreprocessor()
        
        # Load metadata
        self.metadata_df = self._load_metadata()
        
    def _load_metadata(self):
        """Load and process metadata from Excel file."""
        logger.info("Loading metadata from Excel file...")
        
        # Read the sheet with proper headers
        df = pd.read_excel(self.excel_path, sheet_name='Cartridge Case Measurement', header=0)
        
        # Rename columns based on the first row
        df.columns = ['Specimen_Name', 'Creator', 'NIST_Measurement', 'File_Name', 'Measurement_Type', 
                      'Instrument_Brand', 'Instrument_Model', 'Measurand', 'Breech_Face', 'Firing_Pin',
                      'Ejector_Mark', 'Aperture_Shear', 'Lateral_Resolution_um', 'Vertical_Resolution_um',
                      'Lighting_Direction', 'Objective_Magnification', 'Numerical_Aperture', 'Comments']
        
        # Remove the first row which contains the headers
        df = df.iloc[1:].reset_index(drop=True)
        
        # Filter for records with resolution data
        df = df[df['Lateral_Resolution_um'].notna()]
        
        logger.info(f"Loaded {len(df)} records with resolution data")
        return df
    
    def _microns_to_dpi(self, microns_per_pixel):
        """Convert microns per pixel to DPI."""
        if pd.isna(microns_per_pixel) or microns_per_pixel == 0:
            return None
        
        # Convert microns to inches (1 inch = 25400 microns)
        inches_per_pixel = float(microns_per_pixel) / 25400.0
        dpi = 1.0 / inches_per_pixel
        return int(round(dpi))
    
    def _create_image_with_metadata(self, image_path, dpi_value):
        """Create a temporary image with embedded DPI metadata."""
        try:
            # Open the original image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create EXIF data with DPI information
                exif_dict = {
                    "0th": {
                        piexif.ImageIFD.XResolution: (dpi_value, 1),
                        piexif.ImageIFD.YResolution: (dpi_value, 1),
                        piexif.ImageIFD.ResolutionUnit: 2,  # inches
                        piexif.ImageIFD.Software: "NIST Test Suite"
                    }
                }
                
                exif_bytes = piexif.dump(exif_dict)
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                temp_path = temp_file.name
                temp_file.close()
                
                # Save with EXIF data
                img.save(temp_path, "JPEG", exif=exif_bytes, quality=95)
                
                return temp_path
                
        except Exception as e:
            logger.error(f"Error creating image with metadata: {e}")
            return None
    
    def test_calibration_accuracy(self):
        """Test calibration accuracy with real NIST images."""
        logger.info("Testing calibration accuracy with real NIST images...")
        
        results = []
        test_count = 0
        max_tests = 10  # Limit tests for performance
        
        for _, row in self.metadata_df.iterrows():
            if test_count >= max_tests:
                break
                
            file_name = row['File_Name']
            lateral_res_um = row['Lateral_Resolution_um']
            magnification = row['Objective_Magnification']
            
            # Skip if no resolution data
            if pd.isna(lateral_res_um):
                continue
                
            image_path = os.path.join(self.images_dir, file_name)
            
            # Check if image exists
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {file_name}")
                continue
                
            # Convert microns to DPI
            expected_dpi = self._microns_to_dpi(lateral_res_um)
            if expected_dpi is None:
                continue
                
            logger.info(f"Testing {file_name}: {lateral_res_um}μm/pixel → {expected_dpi} DPI")
            
            # Create image with metadata
            temp_image_path = self._create_image_with_metadata(image_path, expected_dpi)
            if temp_image_path is None:
                continue
                
            try:
                # Test calibration from metadata
                calibration_data = self.calibrator.calibrate_from_metadata(temp_image_path)
                
                if calibration_data:
                    actual_dpi = calibration_data.pixels_per_inch
                    error_percent = abs(actual_dpi - expected_dpi) / expected_dpi * 100
                    
                    result = {
                        'file_name': file_name,
                        'expected_dpi': expected_dpi,
                        'actual_dpi': actual_dpi,
                        'error_percent': error_percent,
                        'lateral_res_um': lateral_res_um,
                        'magnification': magnification,
                        'success': True
                    }
                    
                    logger.info(f"  Expected: {expected_dpi} DPI, Got: {actual_dpi} DPI, Error: {error_percent:.2f}%")
                else:
                    result = {
                        'file_name': file_name,
                        'expected_dpi': expected_dpi,
                        'actual_dpi': None,
                        'error_percent': None,
                        'lateral_res_um': lateral_res_um,
                        'magnification': magnification,
                        'success': False
                    }
                    logger.warning(f"  Failed to calibrate {file_name}")
                
                results.append(result)
                test_count += 1
                
            except Exception as e:
                logger.error(f"Error testing {file_name}: {e}")
                
            finally:
                # Clean up temporary file
                if temp_image_path and os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
        
        return results
    
    def test_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline with real images."""
        logger.info("Testing preprocessing pipeline with real NIST images...")
        
        results = []
        test_count = 0
        max_tests = 5  # Limit tests for performance
        
        for _, row in self.metadata_df.iterrows():
            if test_count >= max_tests:
                break
                
            file_name = row['File_Name']
            lateral_res_um = row['Lateral_Resolution_um']
            
            # Skip if no resolution data
            if pd.isna(lateral_res_um):
                continue
                
            image_path = os.path.join(self.images_dir, file_name)
            
            # Check if image exists
            if not os.path.exists(image_path):
                continue
                
            # Convert microns to DPI
            expected_dpi = self._microns_to_dpi(lateral_res_um)
            if expected_dpi is None:
                continue
                
            logger.info(f"Testing preprocessing pipeline with {file_name}")
            
            # Create image with metadata
            temp_image_path = self._create_image_with_metadata(image_path, expected_dpi)
            if temp_image_path is None:
                continue
                
            try:
                # Test preprocessing
                result = self.preprocessor.preprocess_image(temp_image_path)
                
                pipeline_result = {
                    'file_name': file_name,
                    'expected_dpi': expected_dpi,
                    'lateral_res_um': lateral_res_um,
                    'preprocessing_success': result is not None,
                    'steps_applied': len(result.applied_steps) if result else 0,
                    'quality_metrics_count': len(result.quality_metrics) if result else 0,
                    'calibration_data_present': result.calibration_data is not None if result else False,
                    'nist_compliant': result.nist_compliant if result else False,
                    'nist_compliance_report': result.nist_compliance_report is not None if result else False
                }
                
                if result and result.calibration_data:
                    pipeline_result['calibrated_dpi'] = result.calibration_data.pixels_per_inch
                    pipeline_result['dpi_error_percent'] = abs(result.calibration_data.pixels_per_inch - expected_dpi) / expected_dpi * 100
                
                results.append(pipeline_result)
                test_count += 1
                
                logger.info(f"  Preprocessing: {'Success' if pipeline_result['preprocessing_success'] else 'Failed'}")
                logger.info(f"  Steps applied: {pipeline_result['steps_applied']}")
                logger.info(f"  Quality metrics: {pipeline_result['quality_metrics_count']}")
                logger.info(f"  Calibration data: {'Present' if pipeline_result['calibration_data_present'] else 'Missing'}")
                logger.info(f"  NIST compliant: {pipeline_result['nist_compliant']}")
                
                if 'calibrated_dpi' in pipeline_result:
                    logger.info(f"  Calibrated DPI: {pipeline_result['calibrated_dpi']} (error: {pipeline_result['dpi_error_percent']:.2f}%)")
                
            except Exception as e:
                logger.error(f"Error in preprocessing pipeline for {file_name}: {e}")
                
            finally:
                # Clean up temporary file
                if temp_image_path and os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
        
        return results
    
    def test_different_resolutions(self):
        """Test calibration with different resolution values."""
        logger.info("Testing calibration with different resolution values...")
        
        # Get unique resolution values
        unique_resolutions = self.metadata_df['Lateral_Resolution_um'].unique()
        logger.info(f"Found {len(unique_resolutions)} unique resolution values: {sorted(unique_resolutions)}")
        
        results = {}
        
        for resolution in unique_resolutions:
            if pd.isna(resolution):
                continue
                
            dpi = self._microns_to_dpi(resolution)
            if dpi is None:
                continue
                
            # Get a sample image for this resolution
            sample_row = self.metadata_df[self.metadata_df['Lateral_Resolution_um'] == resolution].iloc[0]
            file_name = sample_row['File_Name']
            image_path = os.path.join(self.images_dir, file_name)
            
            if not os.path.exists(image_path):
                continue
                
            logger.info(f"Testing resolution {resolution}μm/pixel ({dpi} DPI) with {file_name}")
            
            # Create image with metadata
            temp_image_path = self._create_image_with_metadata(image_path, dpi)
            if temp_image_path is None:
                continue
                
            try:
                # Test calibration
                calibration_data = self.calibrator.calibrate_from_metadata(temp_image_path)
                
                if calibration_data:
                    actual_dpi = calibration_data.pixels_per_inch
                    error_percent = abs(actual_dpi - dpi) / dpi * 100
                    
                    results[resolution] = {
                        'expected_dpi': dpi,
                        'actual_dpi': actual_dpi,
                        'error_percent': error_percent,
                        'success': True
                    }
                    
                    logger.info(f"  Expected: {dpi} DPI, Got: {actual_dpi} DPI, Error: {error_percent:.2f}%")
                else:
                    results[resolution] = {
                        'expected_dpi': dpi,
                        'actual_dpi': None,
                        'error_percent': None,
                        'success': False
                    }
                    logger.warning(f"  Failed to calibrate")
                
            except Exception as e:
                logger.error(f"Error testing resolution {resolution}: {e}")
                
            finally:
                # Clean up temporary file
                if temp_image_path and os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
        
        return results
    
    def generate_report(self, calibration_results, pipeline_results, resolution_results):
        """Generate a comprehensive test report."""
        logger.info("Generating test report...")
        
        print("\n" + "="*80)
        print("NIST REAL IMAGES CALIBRATION TEST REPORT")
        print("="*80)
        
        # Calibration accuracy results
        print("\n1. CALIBRATION ACCURACY RESULTS:")
        print("-" * 40)
        
        if calibration_results:
            successful_tests = [r for r in calibration_results if r['success']]
            failed_tests = [r for r in calibration_results if not r['success']]
            
            print(f"Total tests: {len(calibration_results)}")
            print(f"Successful: {len(successful_tests)}")
            print(f"Failed: {len(failed_tests)}")
            
            if successful_tests:
                errors = [r['error_percent'] for r in successful_tests]
                print(f"Average error: {np.mean(errors):.2f}%")
                print(f"Max error: {np.max(errors):.2f}%")
                print(f"Min error: {np.min(errors):.2f}%")
                
                print("\nDetailed results:")
                for result in successful_tests:
                    print(f"  {result['file_name']}: {result['expected_dpi']} → {result['actual_dpi']} DPI ({result['error_percent']:.2f}% error)")
        else:
            print("No calibration results available")
        
        # Pipeline results
        print("\n2. PREPROCESSING PIPELINE RESULTS:")
        print("-" * 40)
        
        if pipeline_results:
            successful_preprocessing = [r for r in pipeline_results if r['preprocessing_success']]
            with_calibration = [r for r in pipeline_results if r['calibration_data_present']]
            nist_compliant = [r for r in pipeline_results if r['nist_compliant']]
            
            print(f"Total pipeline tests: {len(pipeline_results)}")
            print(f"Successful preprocessing: {len(successful_preprocessing)}")
            print(f"With calibration data: {len(with_calibration)}")
            print(f"NIST compliant: {len(nist_compliant)}")
            
            if with_calibration:
                dpi_errors = [r['dpi_error_percent'] for r in pipeline_results if 'dpi_error_percent' in r]
                if dpi_errors:
                    print(f"Average DPI error in pipeline: {np.mean(dpi_errors):.2f}%")
            
            print("\nDetailed pipeline results:")
            for result in pipeline_results:
                print(f"  {result['file_name']}:")
                print(f"    Steps: {result['steps_applied']}, Metrics: {result['quality_metrics_count']}")
                print(f"    Calibration: {'Yes' if result['calibration_data_present'] else 'No'}")
                print(f"    NIST compliant: {result['nist_compliant']}")
                if 'calibrated_dpi' in result:
                    print(f"    DPI: {result['calibrated_dpi']} (error: {result['dpi_error_percent']:.2f}%)")
        else:
            print("No pipeline results available")
        
        # Resolution results
        print("\n3. RESOLUTION-SPECIFIC RESULTS:")
        print("-" * 40)
        
        if resolution_results:
            for resolution, result in resolution_results.items():
                if result['success']:
                    print(f"  {resolution}μm/pixel ({result['expected_dpi']} DPI): {result['actual_dpi']} DPI ({result['error_percent']:.2f}% error)")
                else:
                    print(f"  {resolution}μm/pixel ({result['expected_dpi']} DPI): FAILED")
        else:
            print("No resolution-specific results available")
        
        print("\n" + "="*80)
        print("TEST COMPLETED")
        print("="*80)

def main():
    """Main test function."""
    logger.info("Starting NIST real images calibration test...")
    
    try:
        tester = NISTRealImageTester()
        
        # Run tests
        calibration_results = tester.test_calibration_accuracy()
        pipeline_results = tester.test_preprocessing_pipeline()
        resolution_results = tester.test_different_resolutions()
        
        # Generate report
        tester.generate_report(calibration_results, pipeline_results, resolution_results)
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)