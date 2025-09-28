"""
icici_converter.py - SAVE THIS AS icici_converter.py
ICICI specific converter implementation with hardcoded mappings
"""

import pandas as pd
from typing import Dict, List
import logging
from base_converter import BaseBrokerConverter

logger = logging.getLogger(__name__)


class ICICIConverter(BaseBrokerConverter):
    """ICICI Securities specific converter"""
    
    def get_broker_name(self) -> str:
        """Return the broker name"""
        return "ICICI"
    
    def get_column_mappings(self) -> Dict[str, str]:
        """
        Hardcoded mapping of clearing format columns to ICICI input columns
        Output Column -> Input Column
        """
        return {
            'CP Code': 'CP Code',
            'TM Code': 'Broker Code',
            'Scheme': 'Client Name',
            'TM Name': 'Broker Name',
            # 'Instr' is derived, not mapped
            'Symbol': 'Scrip Code',
            'Expiry Dt': 'Expiry',
            'Lot Size': 'No. of Contracts',
            'Strike Price': 'Strike Price',
            'Option Type': 'Call / Put',
            'B/S': 'Buy / Sell',
            'Qty': 'Qty',
            'Lots Traded': 'No. of Contracts',
            'Avg Price': 'Mkt. Rate',
            'Brokerage': 'Pure Brokerage AMT',
            'Taxes': 'Total Taxes'
        }
    
    def get_required_columns(self) -> List[str]:
        """List of required columns in ICICI format"""
        return [
            'CP Code',
            'Broker Code', 
            'Client Name',
            'Broker Name',
            'Scrip Code',
            'Segment Type',
            'Expiry',
            'Strike Price',
            'Call / Put',
            'Buy / Sell',
            'Qty',
            'No. of Contracts',
            'Mkt. Rate',
            'Pure Brokerage AMT',
            'Total Taxes'
        ]
    
    def get_signature_columns(self) -> List[str]:
        """Columns that uniquely identify ICICI format"""
        return [
            'Pure Brokerage AMT',
            'CGST',
            'SGST',
            'Transaction Tax',
            'SEBI Fee',
            'Stt',
            'Settlement Amount'
        ]
    
    def derive_instr_field(self, row: pd.Series) -> str:
        """
        Derive the Instr field based on ICICI logic:
        - If Segment Type contains "Stock" and Call/Put is empty -> FUTSTK
        - If Segment Type contains "Stock" and Call/Put has value -> OPTSTK
        - If Segment Type contains "Index" and Call/Put is empty -> FUTIDX
        - If Segment Type contains "Index" and Call/Put has value -> OPTIDX
        """
        try:
            segment_type = str(row.get('Segment Type', '')).upper()
            call_put = str(row.get('Call / Put', '')).strip()
            
            # Check if Call/Put is empty
            is_option = call_put and call_put.upper() not in ['', 'NAN', 'NONE', 'NULL']
            
            # Determine instrument type based on segment and option flag
            if 'STOCK' in segment_type:
                return 'OPTSTK' if is_option else 'FUTSTK'
            elif 'INDEX' in segment_type or 'IDX' in segment_type:
                return 'OPTIDX' if is_option else 'FUTIDX'
            else:
                # Default fallback
                if is_option:
                    return 'OPTSTK'
                else:
                    return 'FUTSTK'
                    
        except Exception as e:
            logger.warning(f"Error deriving Instr field: {e}")
            return 'FUTSTK'
    
    def convert_row(self, row: pd.Series) -> Dict[str, any]:
        """Override to handle ICICI specific conversions"""
        # Get base conversion
        output = super().convert_row(row)
        
        # ICICI specific adjustments
        
        # Clean up B/S field - ensure it's just B or S
        bs_value = str(output.get('B/S', '')).upper().strip()
        if bs_value.startswith('BUY') or bs_value == 'B':
            output['B/S'] = 'B'
        elif bs_value.startswith('SELL') or bs_value == 'S':
            output['B/S'] = 'S'
        else:
            output['B/S'] = bs_value
        
        # Clean up Option Type - ensure it's CE or PE
        option_type = str(output.get('Option Type', '')).upper().strip()
        if option_type in ['CALL', 'C', 'CE']:
            output['Option Type'] = 'CE'
        elif option_type in ['PUT', 'P', 'PE']:
            output['Option Type'] = 'PE'
        elif not option_type or option_type in ['NAN', 'NONE', 'NULL', '']:
            output['Option Type'] = ''
        
        # Handle lot size - ensure it's numeric
        if 'Lot Size' in output:
            output['Lot Size'] = self.clean_numeric(output['Lot Size'])
            # If lot size is 0, try to use Qty field
            if output['Lot Size'] == 0 and output.get('Qty', 0) > 0:
                # Try to infer lot size
                qty = output['Qty']
                common_lot_sizes = [25, 50, 75, 100, 125, 150, 200, 250, 500, 1000, 1800, 2400, 3000]
                for lot_size in common_lot_sizes:
                    if qty % lot_size == 0:
                        output['Lot Size'] = lot_size
                        output['Lots Traded'] = qty / lot_size
                        break
        
        return output


# Broker Registry - Add new brokers here as they're implemented
BROKER_REGISTRY = {
    'ICICI': ICICIConverter,
    # Future: 'KOTAK': KotakConverter,
    # Future: 'ZERODHA': ZerodhaConverter,
}
