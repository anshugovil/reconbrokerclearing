"""
base_converter.py - SAVE THIS AS base_converter.py
Base converter class that all broker-specific converters will inherit from
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseBrokerConverter(ABC):
    """Abstract base class for all broker converters"""
    
    def __init__(self):
        self.broker_name = self.get_broker_name()
        self.column_mappings = self.get_column_mappings()
        self.required_columns = self.get_required_columns()
        self.signature_columns = self.get_signature_columns()
    
    @abstractmethod
    def get_broker_name(self) -> str:
        """Return the broker name"""
        pass
    
    @abstractmethod
    def get_column_mappings(self) -> Dict[str, str]:
        """Return mapping of output columns to input columns"""
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return list of required input columns"""
        pass
    
    @abstractmethod
    def get_signature_columns(self) -> List[str]:
        """Return columns that uniquely identify this broker format"""
        pass
    
    @abstractmethod
    def derive_instr_field(self, row: pd.Series) -> str:
        """Derive the Instr field based on broker-specific logic"""
        pass
    
    def validate_columns(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate if dataframe has required columns"""
        missing_columns = []
        df_columns_upper = [col.upper().strip() for col in df.columns]
        
        for required_col in self.required_columns:
            if required_col.upper().strip() not in df_columns_upper:
                missing_columns.append(required_col)
        
        is_valid = len(missing_columns) == 0
        return is_valid, missing_columns
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for case-insensitive matching"""
        normalized_map = {}
        for col in df.columns:
            for mapping_col in self.column_mappings.values():
                if mapping_col and col.strip().upper() == mapping_col.strip().upper():
                    normalized_map[col] = mapping_col
                    break
            
            for req_col in self.required_columns:
                if col.strip().upper() == req_col.strip().upper():
                    normalized_map[col] = req_col
                    break
        
        if normalized_map:
            df = df.rename(columns=normalized_map)
        
        return df
    
    def clean_numeric(self, value: Any) -> float:
        """Clean and convert to numeric value"""
        if pd.isna(value) or value == '' or value is None:
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        try:
            cleaned = str(value).replace(',', '').replace('₹', '').replace('$', '').strip()
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0
    
    def clean_date(self, value: Any) -> str:
        """Clean and standardize date format"""
        if pd.isna(value) or value == '' or value is None:
            return ''
        
        try:
            if isinstance(value, datetime):
                return value.strftime('%d-%m-%Y')
            elif isinstance(value, str):
                for fmt in ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%b-%Y', '%d-%b-%y']:
                    try:
                        dt = datetime.strptime(value.strip(), fmt)
                        return dt.strftime('%d-%m-%Y')
                    except:
                        continue
            
            return str(value)
        except:
            return str(value)
    
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert broker format to clearing format"""
        df = self.normalize_column_names(df)
        
        is_valid, missing_cols = self.validate_columns(df)
        if not is_valid:
            raise ValueError(f"Missing required columns for {self.broker_name}: {missing_cols}")
        
        output_data = []
        
        for idx, row in df.iterrows():
            try:
                output_row = self.convert_row(row)
                if output_row:
                    output_data.append(output_row)
            except Exception as e:
                logger.warning(f"Error converting row {idx}: {e}")
                continue
        
        output_df = pd.DataFrame(output_data)
        
        output_columns = [
            'CP Code', 'TM Code', 'Scheme', 'TM Name', 'Instr',
            'Symbol', 'Expiry Dt', 'Lot Size', 'Strike Price',
            'Option Type', 'B/S', 'Qty', 'Lots Traded', 'Avg Price',
            'Brokerage', 'Taxes'
        ]
        
        output_df = output_df[output_columns]
        
        return output_df
    
    def convert_row(self, row: pd.Series) -> Dict[str, Any]:
        """Convert a single row to clearing format"""
        output = {}
        
        for output_col, input_col in self.column_mappings.items():
            if input_col and input_col in row.index:
                value = row[input_col]
                
                if output_col in ['Qty', 'Lot Size', 'Strike Price', 'Avg Price', 'Brokerage', 'Taxes', 'Lots Traded']:
                    output[output_col] = self.clean_numeric(value)
                elif output_col == 'Expiry Dt':
                    output[output_col] = self.clean_date(value)
                else:
                    output[output_col] = str(value) if pd.notna(value) else ''
            else:
                if output_col in ['Qty', 'Lot Size', 'Strike Price', 'Avg Price', 'Brokerage', 'Taxes', 'Lots Traded']:
                    output[output_col] = 0.0
                else:
                    output[output_col] = ''
        
        output['Instr'] = self.derive_instr_field(row)
        
        return output
    
    def detect_match_score(self, df: pd.DataFrame) -> float:
        """Calculate confidence score for broker detection"""
        score = 0.0
        
        df_columns_upper = [col.upper().strip() for col in df.columns]
        signature_matches = 0
        
        for sig_col in self.signature_columns:
            if sig_col.upper().strip() in df_columns_upper:
                signature_matches += 1
        
        if self.signature_columns:
            signature_score = (signature_matches / len(self.signature_columns)) * 60
            score += signature_score
        
        required_matches = 0
        for req_col in self.required_columns:
            if req_col.upper().strip() in df_columns_upper:
                required_matches += 1
        
        if self.required_columns:
            required_score = (required_matches / len(self.required_columns)) * 30
            score += required_score
        
        expected_count = len(self.required_columns)
        actual_count = len(df.columns)
        if abs(actual_count - expected_count) <= 5:
            score += 10
        elif abs(actual_count - expected_count) <= 10:
            score += 5
        
        return score
