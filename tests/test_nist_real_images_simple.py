#!/usr/bin/env python3
"""
Simplified test script for validating spatial calibration with real NIST images.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NISTSimpleCalibrationTester:
    """Simplified test for spatial calibration with real NIST images and metadata."""
    
    def __init__(self):
        self.excel_path = '/home/marco/SIGeC-Balistica/uploads/Muestras NIST FADB/Cary Persistence/StudyInfo.xlsx'
        self.images_dir = '/home/marco/SIGeC-Balistica/uploads/Muestras NIST FADB/Cary Persistence/cc'
        self.calibrator = SpatialCalibrator()
        
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
        max_tests = 15  # Test more images but without heavy preprocessing
        
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
                    actual_dpi = calibration_data.dpi
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
                    actual_dpi = calibration_data.dpi
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
    
    def test_manual_calibration(self):
        """Test manual calibration with known measurements."""
        logger.info("Testing manual calibration with known measurements...")
        
        results = []
        test_count = 0
        max_tests = 5
        
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
                
            # Convert microns to DPI and pixels per mm
            expected_dpi = self._microns_to_dpi(lateral_res_um)
            if expected_dpi is None:
                continue
                
            # Calculate pixels per mm (1mm = 1000 microns)
            pixels_per_mm = 1000.0 / float(lateral_res_um)
            
            logger.info(f"Testing manual calibration for {file_name}")
            logger.info(f"  Expected: {expected_dpi} DPI, {pixels_per_mm:.2f} pixels/mm")
            
            try:
                # Test manual calibration with known pixel distance
                # Assume a 10mm reference distance for testing
                reference_distance_mm = 10.0
                pixel_distance = reference_distance_mm * pixels_per_mm
                
                calibration_data = self.calibrator.calibrate_manual(
                    pixel_distance=pixel_distance,
                    real_distance_mm=reference_distance_mm
                )
                
                if calibration_data:
                    actual_dpi = calibration_data.dpi
                    actual_pixels_per_mm = calibration_data.pixels_per_mm
                    
                    dpi_error = abs(actual_dpi - expected_dpi) / expected_dpi * 100
                    ppm_error = abs(actual_pixels_per_mm - pixels_per_mm) / pixels_per_mm * 100
                    
                    result = {
                        'file_name': file_name,
                        'expected_dpi': expected_dpi,
                        'actual_dpi': actual_dpi,
                        'dpi_error_percent': dpi_error,
                        'expected_pixels_per_mm': pixels_per_mm,
                        'actual_pixels_per_mm': actual_pixels_per_mm,
                        'ppm_error_percent': ppm_error,
                        'success': True
                    }
                    
                    logger.info(f"  DPI: {expected_dpi} → {actual_dpi} ({dpi_error:.2f}% error)")
                    logger.info(f"  Pixels/mm: {pixels_per_mm:.2f} → {actual_pixels_per_mm:.2f} ({ppm_error:.2f}% error)")
                else:
                    result = {
                        'file_name': file_name,
                        'expected_dpi': expected_dpi,
                        'success': False
                    }
                    logger.warning(f"  Failed manual calibration")
                
                results.append(result)
                test_count += 1
                
            except Exception as e:
                logger.error(f"Error in manual calibration for {file_name}: {e}")
        
        return results
    
    def generate_report(self, calibration_results, resolution_results, manual_results):
        """Generate a comprehensive test report."""
        logger.info("Generating test report...")
        
        print("\n" + "="*80)
        print("NIST REAL IMAGES CALIBRATION TEST REPORT (SIMPLIFIED)")
        print("="*80)
        
        # Calibration accuracy results
        print("\n1. AUTOMATIC CALIBRATION ACCURACY RESULTS:")
        print("-" * 50)
        
        if calibration_results:
            successful_tests = [r for r in calibration_results if r['success']]
            failed_tests = [r for r in calibration_results if not r['success']]
            
            print(f"Total tests: {len(calibration_results)}")
            print(f"Successful: {len(successful_tests)}")
            print(f"Failed: {len(failed_tests)}")
            print(f"Success rate: {len(successful_tests)/len(calibration_results)*100:.1f}%")
            
            if successful_tests:
                errors = [r['error_percent'] for r in successful_tests]
                print(f"Average error: {np.mean(errors):.2f}%")
                print(f"Max error: {np.max(errors):.2f}%")
                print(f"Min error: {np.min(errors):.2f}%")
                print(f"Standard deviation: {np.std(errors):.2f}%")
                
                print("\nDetailed results:")
                for result in successful_tests:
                    print(f"  {result['file_name']}: {result['expected_dpi']} → {result['actual_dpi']} DPI ({result['error_percent']:.2f}% error)")
                    
                # Check accuracy thresholds
                high_accuracy = len([r for r in successful_tests if r['error_percent'] < 1.0])
                good_accuracy = len([r for r in successful_tests if r['error_percent'] < 5.0])
                
                print(f"\nAccuracy analysis:")
                print(f"  High accuracy (<1% error): {high_accuracy}/{len(successful_tests)} ({high_accuracy/len(successful_tests)*100:.1f}%)")
                print(f"  Good accuracy (<5% error): {good_accuracy}/{len(successful_tests)} ({good_accuracy/len(successful_tests)*100:.1f}%)")
        else:
            print("No calibration results available")
        
        # Resolution results
        print("\n2. RESOLUTION-SPECIFIC RESULTS:")
        print("-" * 40)
        
        if resolution_results:
            print("Resolution testing results:")
            for resolution, result in sorted(resolution_results.items()):
                if result['success']:
                    print(f"  {resolution}μm/pixel ({result['expected_dpi']} DPI): {result['actual_dpi']} DPI ({result['error_percent']:.2f}% error)")
                else:
                    print(f"  {resolution}μm/pixel ({result['expected_dpi']} DPI): FAILED")
        else:
            print("No resolution-specific results available")
        
        # Manual calibration results
        print("\n3. MANUAL CALIBRATION RESULTS:")
        print("-" * 40)
        
        if manual_results:
            successful_manual = [r for r in manual_results if r['success']]
            
            print(f"Manual calibration tests: {len(manual_results)}")
            print(f"Successful: {len(successful_manual)}")
            
            if successful_manual:
                dpi_errors = [r['dpi_error_percent'] for r in successful_manual]
                ppm_errors = [r['ppm_error_percent'] for r in successful_manual]
                
                print(f"Average DPI error: {np.mean(dpi_errors):.2f}%")
                print(f"Average pixels/mm error: {np.mean(ppm_errors):.2f}%")
                
                print("\nDetailed manual calibration results:")
                for result in successful_manual:
                    print(f"  {result['file_name']}:")
                    print(f"    DPI: {result['expected_dpi']} → {result['actual_dpi']} ({result['dpi_error_percent']:.2f}% error)")
                    print(f"    Pixels/mm: {result['expected_pixels_per_mm']:.2f} → {result['actual_pixels_per_mm']:.2f} ({result['ppm_error_percent']:.2f}% error)")
        else:
            print("No manual calibration results available")
        
        # Summary
        print("\n4. SUMMARY:")
        print("-" * 20)
        
        total_tests = len(calibration_results) if calibration_results else 0
        successful_auto = len([r for r in calibration_results if r['success']]) if calibration_results else 0
        successful_manual = len([r for r in manual_results if r['success']]) if manual_results else 0
        
        print(f"Total automatic calibration tests: {total_tests}")
        print(f"Successful automatic calibrations: {successful_auto}")
        print(f"Successful manual calibrations: {successful_manual}")
        
        if successful_auto > 0:
            avg_error = np.mean([r['error_percent'] for r in calibration_results if r['success']])
            print(f"Overall average calibration error: {avg_error:.2f}%")
            
            if avg_error < 2.0:
                print("✅ EXCELLENT: Calibration system shows high accuracy")
            elif avg_error < 5.0:
                print("✅ GOOD: Calibration system shows acceptable accuracy")
            else:
                print("⚠️  WARNING: Calibration system may need improvement")
        
        print("\n" + "="*80)
        print("TEST COMPLETED")
        print("="*80)

def main():
    """Main test function."""
    logger.info("Starting simplified NIST real images calibration test...")
    
    try:
        tester = NISTSimpleCalibrationTester()
        
        # Run tests
        calibration_results = tester.test_calibration_accuracy()
        resolution_results = tester.test_different_resolutions()
        manual_results = tester.test_manual_calibration()
        
        # Generate report
        tester.generate_report(calibration_results, resolution_results, manual_results)
        
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