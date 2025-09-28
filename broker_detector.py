"""
broker_detector.py - SAVE THIS AS broker_detector.py
Auto-detect broker type from file content and structure
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
from icici_converter import BROKER_REGISTRY

logger = logging.getLogger(__name__)


class BrokerDetector:
    """Automatically detect broker type from file"""
    
    def __init__(self):
        self.broker_converters = {
            name: converter_class() 
            for name, converter_class in BROKER_REGISTRY.items()
        }
        
    def detect_broker(self, df: pd.DataFrame, filename: str = "") -> Tuple[str, float, Dict]:
        """
        Detect broker type from dataframe content
        Returns: (broker_name, confidence_score, detection_details)
        """
        detection_results = {}
        
        # Method 1: Check each registered broker's signature
        for broker_name, converter in self.broker_converters.items():
            score = converter.detect_match_score(df)
            detection_results[broker_name] = {
                'signature_score': score,
                'name_in_file': 0,
                'name_in_filename': 0,
                'total_score': score
            }
        
        # Method 2: Look for broker name in the dataframe
        broker_name_found = self._find_broker_name_in_content(df)
        if broker_name_found:
            for broker in detection_results:
                if broker.upper() in broker_name_found.upper():
                    detection_results[broker]['name_in_file'] = 30
                    detection_results[broker]['total_score'] += 30
        
        # Method 3: Check filename for broker name
        if filename:
            filename_upper = filename.upper()
            for broker in detection_results:
                if broker.upper() in filename_upper:
                    detection_results[broker]['name_in_filename'] = 20
                    detection_results[broker]['total_score'] += 20
        
        # Find best match
        best_broker = None
        best_score = 0
        best_details = {}
        
        for broker, details in detection_results.items():
            if details['total_score'] > best_score:
                best_score = details['total_score']
                best_broker = broker
                best_details = details
        
        # Log detection results
        logger.info(f"Detection results for {filename}:")
        for broker, details in detection_results.items():
            logger.info(f"  {broker}: {details['total_score']:.1f} "
                       f"(signature: {details['signature_score']:.1f}, "
                       f"in_file: {details['name_in_file']}, "
                       f"in_name: {details['name_in_filename']})")
        
        return best_broker, best_score, best_details
    
    def _find_broker_name_in_content(self, df: pd.DataFrame) -> Optional[str]:
        """Look for broker name in file content"""
        try:
            # Check if there's a 'Broker Name' column
            for col in df.columns:
                if 'BROKER' in col.upper() and 'NAME' in col.upper():
                    # Get unique values from this column
                    values = df[col].dropna().unique()
                    for val in values:
                        val_str = str(val).upper()
                        # Check against known brokers
                        for broker in self.broker_converters.keys():
                            if broker.upper() in val_str:
                                return broker
            
            # Check first few rows for broker information
            if len(df) > 0:
                # Convert first 5 rows to string and search
                first_rows_text = df.head(5).to_string().upper()
                
                # Known broker name patterns
                broker_patterns = {
                    'ICICI': ['ICICI', 'ICICI SECURITIES', 'ISEC'],
                    'KOTAK': ['KOTAK', 'KOTAK SECURITIES', 'KSL'],
                    'ZERODHA': ['ZERODHA', 'ZERODHA BROKING'],
                    # Add more as needed
                }
                
                for broker, patterns in broker_patterns.items():
                    for pattern in patterns:
                        if pattern in first_rows_text:
                            return broker
                            
        except Exception as e:
            logger.debug(f"Error searching for broker name: {e}")
        
        return None
    
    def validate_detection(self, df: pd.DataFrame, broker_name: str) -> Tuple[bool, List[str]]:
        """Validate if the detected broker is correct by checking required columns"""
        if broker_name not in self.broker_converters:
            return False, [f"Unknown broker: {broker_name}"]
        
        converter = self.broker_converters[broker_name]
        is_valid, missing_cols = converter.validate_columns(df)
        
        return is_valid, missing_cols
    
    def process_file(self, file_path: str, detected_broker: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], str, Dict]:
        """
        Process a file with auto-detection or specified broker
        Returns: (converted_df, broker_used, processing_info)
        """
        try:
            # Read file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Detect broker if not specified
            if not detected_broker:
                detected_broker, confidence, details = self.detect_broker(df, file_path)
                
                if not detected_broker or confidence < 30:
                    return None, "", {
                        'error': 'Could not detect broker type',
                        'confidence': confidence,
                        'details': details
                    }
            else:
                confidence = 100
                details = {'manual_selection': True}
            
            # Validate detection
            is_valid, missing_cols = self.validate_detection(df, detected_broker)
            
            if not is_valid:
                return None, detected_broker, {
                    'error': f'Missing columns for {detected_broker}',
                    'missing_columns': missing_cols,
                    'confidence': confidence
                }
            
            # Convert using detected broker
            converter = self.broker_converters[detected_broker]
            converted_df = converter.convert(df)
            
            return converted_df, detected_broker, {
                'success': True,
                'rows_processed': len(converted_df),
                'confidence': confidence,
                'details': details
            }
            
        except Exception as e:
            return None, "", {
                'error': str(e),
                'confidence': 0
            }


def get_available_brokers() -> List[str]:
    """Get list of available broker converters"""
    return list(BROKER_REGISTRY.keys())
