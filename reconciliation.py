"""
reconciliation.py - SAVE THIS AS reconciliation.py
Reconciliation module to compare converted files with clearing broker files
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TradeReconciliation:
    """Reconcile converted broker files with clearing broker files"""
    
    def __init__(self):
        self.match_fields = [
            'CP Code', 'Symbol', 'Expiry Dt', 'Instr', 
            'Qty', 'Lots Traded', 'Avg Price'
        ]
        self.tolerance = {
            'Qty': 0.01,  # Allow 0.01 difference in quantity
            'Lots Traded': 0.01,
            'Avg Price': 0.01  # Allow 0.01 difference in price
        }
    
    def create_match_key(self, row: pd.Series) -> str:
        """Create a unique key for matching trades"""
        # Create key from CP Code, Symbol, Expiry, Instr
        cp_code = str(row.get('CP Code', '')).strip()
        symbol = str(row.get('Symbol', '')).strip()
        expiry = str(row.get('Expiry Dt', '')).strip()
        instr = str(row.get('Instr', '')).strip()
        
        return f"{cp_code}|{symbol}|{expiry}|{instr}"
    
    def compare_numeric_fields(self, val1: float, val2: float, tolerance: float = 0.01) -> bool:
        """Compare numeric fields with tolerance"""
        try:
            v1 = float(val1) if pd.notna(val1) else 0.0
            v2 = float(val2) if pd.notna(val2) else 0.0
            return abs(v1 - v2) <= tolerance
        except:
            return str(val1) == str(val2)
    
    def reconcile(self, converted_df: pd.DataFrame, clearing_df: pd.DataFrame) -> Dict:
        """
        Perform reconciliation between converted and clearing broker files
        Returns detailed reconciliation results
        """
        results = {
            'summary': {},
            'matched_trades': [],
            'unmatched_converted': [],
            'unmatched_clearing': [],
            'mismatched_values': [],
            'details': []
        }
        
        # Ensure required columns exist
        for df, name in [(converted_df, 'Converted'), (clearing_df, 'Clearing')]:
            missing_cols = [col for col in self.match_fields if col not in df.columns]
            if missing_cols:
                logger.warning(f"{name} file missing columns: {missing_cols}")
                # Handle Symbol vs Scrip Code naming
                if 'Symbol' not in df.columns and 'Scrip Code' in df.columns:
                    df['Symbol'] = df['Scrip Code']
        
        # Create copies to work with
        conv_df = converted_df.copy()
        clear_df = clearing_df.copy()
        
        # Add match keys
        conv_df['match_key'] = conv_df.apply(self.create_match_key, axis=1)
        clear_df['match_key'] = clear_df.apply(self.create_match_key, axis=1)
        
        # Add row identifiers
        conv_df['source_row'] = range(len(conv_df))
        clear_df['source_row'] = range(len(clear_df))
        
        # Track matched rows
        matched_conv_rows = set()
        matched_clear_rows = set()
        
        # Group by match key for efficient matching
        conv_groups = conv_df.groupby('match_key')
        clear_groups = clear_df.groupby('match_key')
        
        # Process each unique trade in converted file
        for match_key, conv_group in conv_groups:
            if match_key in clear_groups.groups:
                clear_group = clear_df.loc[clear_groups.groups[match_key]]
                
                # Try to match trades within groups
                for conv_idx, conv_row in conv_group.iterrows():
                    if conv_idx in matched_conv_rows:
                        continue
                    
                    for clear_idx, clear_row in clear_group.iterrows():
                        if clear_idx in matched_clear_rows:
                            continue
                        
                        # Check if numeric fields match within tolerance
                        qty_match = self.compare_numeric_fields(
                            conv_row['Qty'], clear_row['Qty'], self.tolerance['Qty']
                        )
                        lots_match = self.compare_numeric_fields(
                            conv_row['Lots Traded'], clear_row['Lots Traded'], self.tolerance['Lots Traded']
                        )
                        price_match = self.compare_numeric_fields(
                            conv_row['Avg Price'], clear_row['Avg Price'], self.tolerance['Avg Price']
                        )
                        
                        if qty_match and lots_match and price_match:
                            # Perfect match found
                            matched_conv_rows.add(conv_idx)
                            matched_clear_rows.add(clear_idx)
                            
                            results['matched_trades'].append({
                                'converted_row': conv_row['source_row'],
                                'clearing_row': clear_row['source_row'],
                                'CP Code': conv_row['CP Code'],
                                'Symbol': conv_row['Symbol'],
                                'Expiry Dt': conv_row['Expiry Dt'],
                                'Instr': conv_row['Instr'],
                                'Qty': conv_row['Qty'],
                                'Avg Price': conv_row['Avg Price']
                            })
                            break
                        else:
                            # Partial match - same trade but different values
                            mismatches = {}
                            if not qty_match:
                                mismatches['Qty'] = {
                                    'converted': conv_row['Qty'],
                                    'clearing': clear_row['Qty'],
                                    'difference': abs(float(conv_row['Qty']) - float(clear_row['Qty']))
                                }
                            if not lots_match:
                                mismatches['Lots Traded'] = {
                                    'converted': conv_row['Lots Traded'],
                                    'clearing': clear_row['Lots Traded'],
                                    'difference': abs(float(conv_row['Lots Traded']) - float(clear_row['Lots Traded']))
                                }
                            if not price_match:
                                mismatches['Avg Price'] = {
                                    'converted': conv_row['Avg Price'],
                                    'clearing': clear_row['Avg Price'],
                                    'difference': abs(float(conv_row['Avg Price']) - float(clear_row['Avg Price']))
                                }
                            
                            if mismatches:
                                matched_conv_rows.add(conv_idx)
                                matched_clear_rows.add(clear_idx)
                                
                                results['mismatched_values'].append({
                                    'converted_row': conv_row['source_row'],
                                    'clearing_row': clear_row['source_row'],
                                    'CP Code': conv_row['CP Code'],
                                    'Symbol': conv_row['Symbol'],
                                    'Expiry Dt': conv_row['Expiry Dt'],
                                    'Instr': conv_row['Instr'],
                                    'mismatches': mismatches
                                })
                                break
        
        # Find unmatched trades in converted file
        for idx, row in conv_df.iterrows():
            if idx not in matched_conv_rows:
                results['unmatched_converted'].append({
                    'row': row['source_row'],
                    'CP Code': row['CP Code'],
                    'Symbol': row['Symbol'],
                    'Expiry Dt': row['Expiry Dt'],
                    'Instr': row['Instr'],
                    'Qty': row['Qty'],
                    'Avg Price': row['Avg Price'],
                    'Brokerage': row.get('Brokerage', 0),
                    'Taxes': row.get('Taxes', 0)
                })
        
        # Find unmatched trades in clearing file
        for idx, row in clear_df.iterrows():
            if idx not in matched_clear_rows:
                results['unmatched_clearing'].append({
                    'row': row['source_row'],
                    'CP Code': row['CP Code'],
                    'Symbol': row['Symbol'],
                    'Expiry Dt': row['Expiry Dt'],
                    'Instr': row['Instr'],
                    'Qty': row['Qty'],
                    'Avg Price': row['Avg Price']
                })
        
        # Calculate summary statistics
        results['summary'] = {
            'total_converted_trades': len(conv_df),
            'total_clearing_trades': len(clear_df),
            'matched_trades': len(results['matched_trades']),
            'mismatched_values': len(results['mismatched_values']),
            'unmatched_converted': len(results['unmatched_converted']),
            'unmatched_clearing': len(results['unmatched_clearing']),
            'match_rate': (len(results['matched_trades']) / len(conv_df) * 100) if len(conv_df) > 0 else 0
        }
        
        return results
    
    def generate_recon_report(self, results: Dict) -> pd.DataFrame:
        """Generate a detailed reconciliation report as DataFrame"""
        report_data = []
        
        # Add matched trades
        for trade in results['matched_trades']:
            report_data.append({
                'Status': '✅ Matched',
                'Type': 'Perfect Match',
                'Conv Row': trade['converted_row'],
                'Clear Row': trade['clearing_row'],
                'CP Code': trade['CP Code'],
                'Symbol': trade['Symbol'],
                'Expiry': trade['Expiry Dt'],
                'Instr': trade['Instr'],
                'Qty': trade['Qty'],
                'Price': trade['Avg Price'],
                'Issue': ''
            })
        
        # Add mismatched values
        for trade in results['mismatched_values']:
            issues = []
            for field, mismatch in trade['mismatches'].items():
                issues.append(f"{field}: Conv={mismatch['converted']}, Clear={mismatch['clearing']}")
            
            report_data.append({
                'Status': '⚠️ Mismatch',
                'Type': 'Value Mismatch',
                'Conv Row': trade['converted_row'],
                'Clear Row': trade['clearing_row'],
                'CP Code': trade['CP Code'],
                'Symbol': trade['Symbol'],
                'Expiry': trade['Expiry Dt'],
                'Instr': trade['Instr'],
                'Qty': trade.get('Qty', ''),
                'Price': trade.get('Avg Price', ''),
                'Issue': '; '.join(issues)
            })
        
        # Add unmatched converted trades
        for trade in results['unmatched_converted']:
            report_data.append({
                'Status': '❌ Unmatched',
                'Type': 'Only in Converted',
                'Conv Row': trade['row'],
                'Clear Row': '',
                'CP Code': trade['CP Code'],
                'Symbol': trade['Symbol'],
                'Expiry': trade['Expiry Dt'],
                'Instr': trade['Instr'],
                'Qty': trade['Qty'],
                'Price': trade['Avg Price'],
                'Issue': 'Trade not found in clearing file'
            })
        
        # Add unmatched clearing trades
        for trade in results['unmatched_clearing']:
            report_data.append({
                'Status': '❌ Unmatched',
                'Type': 'Only in Clearing',
                'Conv Row': '',
                'Clear Row': trade['row'],
                'CP Code': trade['CP Code'],
                'Symbol': trade['Symbol'],
                'Expiry': trade['Expiry Dt'],
                'Instr': trade['Instr'],
                'Qty': trade['Qty'],
                'Price': trade['Avg Price'],
                'Issue': 'Trade not found in converted file'
            })
        
        # Create DataFrame and sort
        report_df = pd.DataFrame(report_data)
        if not report_df.empty:
            # Sort by Status (Matched first, then Mismatch, then Unmatched), then by CP Code and Symbol
            status_order = {'✅ Matched': 0, '⚠️ Mismatch': 1, '❌ Unmatched': 2}
            report_df['sort_order'] = report_df['Status'].map(status_order)
            report_df = report_df.sort_values(['sort_order', 'CP Code', 'Symbol'])
            report_df = report_df.drop('sort_order', axis=1)
        
        return report_df
