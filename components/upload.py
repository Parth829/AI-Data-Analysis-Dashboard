import streamlit as st
import pandas as pd
import io

def render_upload_section():
    """
    Renders the file upload section and returns the loaded DataFrame.
    Returns:
        pd.DataFrame or None if no file is uploaded.
    """
    st.header("1. Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", 
        type=["csv", "xlsx", "xls"],
        help="Upload your dataset to get started. Max file size depends on your Streamlit configuration."
    )
    
    if uploaded_file is not None:
        try:
            # Determine file type and read
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            with st.spinner("Loading dataset..."):
                if file_extension == 'csv':
                    try:
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin1')
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format!")
                    return None
            
            # Show preview
            st.subheader("Dataset Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", f"{df.shape[1]:,}")
                
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show column types
            with st.expander("View Column Data Types"):
                dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).astype(str)
                st.dataframe(dtypes_df, use_container_width=True)
                
            return df
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
            
    return None
