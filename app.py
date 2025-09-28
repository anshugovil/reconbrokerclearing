"""
app.py - SAVE THIS AS app.py
Main Streamlit application for Broker Format Converter
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64
from typing import List, Optional

# Import modules - make sure these files are in the same directory
try:
    from broker_detector import BrokerDetector, get_available_brokers
except ImportError:
    st.error("Error: Could not import broker_detector.py. Make sure all module files are in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Broker Format Converter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = {}

# Title and description
st.title("🔄 Executing to Clearing Broker Format Converter")
st.markdown("### Intelligent Multi-Broker File Processing System")
st.markdown("---")

# Sidebar for instructions and settings
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    ### How to use:
    1. **Upload files** - Drop one or multiple broker files
    2. **Auto-detection** - System identifies broker type
    3. **Review & Override** - Check detection results
    4. **Convert** - Process all files
    5. **Download** - Get converted CSV files
    
    ### Features:
    - ✅ Auto-detects broker format
    - ✅ Handles multiple files at once
    - ✅ ICICI format supported
    - ✅ More brokers coming soon
    
    ### Confidence Levels:
    - 🟢 **High (>70%)** - Auto-process
    - 🟡 **Medium (40-70%)** - Review recommended
    - 🔴 **Low (<40%)** - Manual selection needed
    """)
    
    st.markdown("---")
    st.header("⚙️ Settings")
    
    # Auto-process threshold
    auto_process_threshold = st.slider(
        "Auto-process confidence threshold (%)",
        min_value=50,
        max_value=90,
        value=70,
        step=5,
        help="Files with confidence above this will be auto-processed"
    )
    
    # Show detailed logs
    show_logs = st.checkbox("Show detailed processing logs", value=False)
    
    # Available brokers
    st.markdown("---")
    st.header("🏦 Available Brokers")
    available_brokers = get_available_brokers()
    for broker in available_brokers:
        st.success(f"✓ {broker}")
    
    st.caption(f"Total: {len(available_brokers)} broker(s) configured")

# Main content area
def get_confidence_badge(confidence: float) -> str:
    """Get colored badge for confidence level"""
    if confidence >= 70:
        return f"🟢 {confidence:.0f}%"
    elif confidence >= 40:
        return f"🟡 {confidence:.0f}%"
    else:
        return f"🔴 {confidence:.0f}%"

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create a download link for dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'

# File upload section
st.header("📁 File Upload")
uploaded_files = st.file_uploader(
    "Upload broker files (Excel or CSV)",
    type=['csv', 'xlsx', 'xls'],
    accept_multiple_files=True,
    help="You can upload multiple files from different brokers"
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s)")
    
    # Detection Phase
    st.header("🔍 Broker Detection")
    
    detector = BrokerDetector()
    detection_container = st.container()
    
    with detection_container:
        detection_data = []
        
        # Progress bar for detection
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {uploaded_file.name}...")
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            try:
                # Read file into dataframe
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Detect broker
                detected_broker, confidence, details = detector.detect_broker(df, uploaded_file.name)
                
                detection_data.append({
                    'File': uploaded_file.name,
                    'Detected Broker': detected_broker or 'Unknown',
                    'Confidence': confidence,
                    'Confidence Badge': get_confidence_badge(confidence),
                    'Rows': len(df),
                    'Columns': len(df.columns),
                    'Status': '✅ Ready' if confidence >= auto_process_threshold else '⚠️ Review needed',
                    'df': df,
                    'file_obj': uploaded_file
                })
                
                # Store in session state
                st.session_state.detection_results[uploaded_file.name] = {
                    'broker': detected_broker,
                    'confidence': confidence,
                    'details': details,
                    'df': df
                }
                
            except Exception as e:
                detection_data.append({
                    'File': uploaded_file.name,
                    'Detected Broker': 'Error',
                    'Confidence': 0,
                    'Confidence Badge': '❌ Error',
                    'Rows': 0,
                    'Columns': 0,
                    'Status': f'❌ {str(e)}',
                    'df': None,
                    'file_obj': uploaded_file
                })
        
        progress_bar.progress(1.0)
        status_text.text("Detection complete!")
        
        # Display detection results
        detection_df = pd.DataFrame(detection_data)
        
        # Create columns for the detection results table
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Detection Results")
            
            # Display simplified table
            display_df = detection_df[['File', 'Detected Broker', 'Confidence Badge', 'Rows', 'Status']].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Summary")
            total_files = len(detection_data)
            ready_files = sum(1 for d in detection_data if d['Confidence'] >= auto_process_threshold)
            
            st.metric("Total Files", total_files)
            st.metric("Ready to Process", ready_files, delta=f"{ready_files/total_files*100:.0f}%")
            
            # Broker breakdown
            broker_counts = detection_df['Detected Broker'].value_counts()
            st.caption("Files by Broker:")
            for broker, count in broker_counts.items():
                if broker != 'Unknown' and broker != 'Error':
                    st.caption(f"• {broker}: {count}")
    
    # Manual override section for low confidence detections
    st.header("🔧 Manual Override")
    
    needs_review = detection_df[detection_df['Confidence'] < auto_process_threshold]
    
    if len(needs_review) > 0:
        st.warning(f"⚠️ {len(needs_review)} file(s) need review")
        
        override_mapping = {}
        for idx, row in needs_review.iterrows():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.text(f"📄 {row['File']}")
            
            with col2:
                st.text(f"Detected: {row['Detected Broker']} ({row['Confidence']:.0f}%)")
            
            with col3:
                manual_broker = st.selectbox(
                    "Override:",
                    options=['Auto'] + get_available_brokers(),
                    key=f"override_{idx}"
                )
                override_mapping[row['File']] = manual_broker
        
        # Store overrides in session state
        st.session_state.overrides = override_mapping
    else:
        st.success("✅ All files detected with high confidence!")
        st.session_state.overrides = {}
    
    # Processing section
    st.header("⚙️ Process Files")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        process_btn = st.button("🚀 Convert All Files", type="primary", use_container_width=True)
    
    with col2:
        merge_files = st.checkbox("Merge into single file", value=False)
    
    if process_btn:
        st.header("📊 Processing Results")
        
        processed_data = []
        all_converted_dfs = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in detection_df.iterrows():
            filename = row['File']
            status_text.text(f"Processing {filename}...")
            progress_bar.progress((idx + 1) / len(detection_df))
            
            try:
                # Determine which broker to use
                if filename in st.session_state.overrides and st.session_state.overrides[filename] != 'Auto':
                    broker_to_use = st.session_state.overrides[filename]
                else:
                    broker_to_use = row['Detected Broker']
                
                if broker_to_use and broker_to_use not in ['Unknown', 'Error']:
                    # Get the converter
                    converter = detector.broker_converters[broker_to_use]
                    
                    # Convert the dataframe
                    df_to_convert = row['df']
                    converted_df = converter.convert(df_to_convert)
                    
                    # Add source column if merging
                    if merge_files:
                        converted_df['Source_File'] = filename
                        converted_df['Source_Broker'] = broker_to_use
                    
                    all_converted_dfs.append(converted_df)
                    
                    processed_data.append({
                        'File': filename,
                        'Broker': broker_to_use,
                        'Input Rows': len(df_to_convert),
                        'Output Rows': len(converted_df),
                        'Status': '✅ Success',
                        'converted_df': converted_df
                    })
                    
                else:
                    processed_data.append({
                        'File': filename,
                        'Broker': 'Unknown',
                        'Input Rows': len(row['df']) if row['df'] is not None else 0,
                        'Output Rows': 0,
                        'Status': '❌ Could not process',
                        'converted_df': None
                    })
                    
            except Exception as e:
                processed_data.append({
                    'File': filename,
                    'Broker': broker_to_use if 'broker_to_use' in locals() else 'Unknown',
                    'Input Rows': len(row['df']) if row['df'] is not None else 0,
                    'Output Rows': 0,
                    'Status': f'❌ Error: {str(e)}',
                    'converted_df': None
                })
                
                if show_logs:
                    st.error(f"Error processing {filename}: {str(e)}")
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Display processing results
        st.subheader("Processing Summary")
        
        process_df = pd.DataFrame(processed_data)
        display_process_df = process_df[['File', 'Broker', 'Input Rows', 'Output Rows', 'Status']]
        st.dataframe(display_process_df, use_container_width=True, hide_index=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            successful = sum(1 for d in processed_data if d['Status'].startswith('✅'))
            st.metric("Successful", f"{successful}/{len(processed_data)}")
        
        with col2:
            total_input_rows = sum(d['Input Rows'] for d in processed_data)
            st.metric("Total Input Rows", total_input_rows)
        
        with col3:
            total_output_rows = sum(d['Output Rows'] for d in processed_data)
            st.metric("Total Output Rows", total_output_rows)
        
        # Download section
        st.header("📥 Download Results")
        
        if merge_files and len(all_converted_dfs) > 0:
            # Merge all dataframes
            merged_df = pd.concat(all_converted_dfs, ignore_index=True)
            
            # Create download
            csv_data = merged_df.to_csv(index=False)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.download_button(
                    label="📥 Download Merged CSV",
                    data=csv_data,
                    file_name=f"clearing_format_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.info(f"Merged file contains {len(merged_df)} rows from {len(all_converted_dfs)} files")
            
            # Preview
            with st.expander("Preview Merged Data"):
                st.dataframe(merged_df.head(50), use_container_width=True)
        
        else:
            # Individual downloads
            st.info("Download individual converted files:")
            
            for data in processed_data:
                if data['converted_df'] is not None:
                    csv_data = data['converted_df'].to_csv(index=False)
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.text(f"📄 {data['File']}")
                    
                    with col2:
                        st.text(f"{data['Output Rows']} rows")
                    
                    with col3:
                        output_filename = data['File'].rsplit('.', 1)[0] + '_clearing_format.csv'
                        st.download_button(
                            label="Download",
                            data=csv_data,
                            file_name=output_filename,
                            mime="text/csv",
                            key=f"download_{data['File']}"
                        )

else:
    # No files uploaded - show sample format
    st.info("👆 Upload broker files to begin")
    
    with st.expander("📋 View Expected ICICI Format"):
        st.markdown("""
        **Required columns for ICICI format:**
        - Exchange, Segment Type, Broker Code, Broker Name
        - Trade Date, Cont. Ref., CP Code, Client Name
        - Scrip Code, Scrip Name, Strike Price, Call / Put
        - Expiry, Buy / Sell, No. of Contracts, Qty
        - Mkt. Rate, Gross Amount, Brokerage Amount
        - Pure Brokerage AMT, Total Taxes
        - CGST, SGST, Stamp Duty, Transaction Tax
        - SEBI Fee, Stt, Settlement Amount
        """)

# Footer
st.markdown("---")
st.caption("Broker Format Converter v1.0 | Support for more brokers coming soon")
