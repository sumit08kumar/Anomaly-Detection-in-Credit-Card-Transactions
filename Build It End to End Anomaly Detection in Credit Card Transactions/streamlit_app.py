import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.anomaly_detector import AnomalyDetector

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .normal-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

def load_detector():
    """Load the trained anomaly detector"""
    try:
        detector = AnomalyDetector()
        detector.load_models('models/trained_models.pkl')
        return detector
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def load_sample_data():
    """Load sample credit card data"""
    try:
        df = pd.read_csv('data/creditcard.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_transaction_plot(df, predictions, scores, pca_components):
    """Create interactive plot of transactions"""
    # Create a copy of the dataframe for plotting
    plot_df = df.copy()
    plot_df['Anomaly'] = predictions
    plot_df['Anomaly_Score'] = scores
    plot_df['PCA1'] = pca_components[:, 0]
    plot_df['PCA2'] = pca_components[:, 1]
    plot_df['Transaction_Type'] = ['Anomaly' if x == 1 else 'Normal' for x in predictions]
    
    # Create scatter plot
    fig = px.scatter(
        plot_df, 
        x='PCA1', 
        y='PCA2',
        color='Transaction_Type',
        hover_data=['Amount', 'Time', 'Anomaly_Score'],
        title="Transaction Visualization (PCA Components)",
        color_discrete_map={'Normal': '#4CAF50', 'Anomaly': '#F44336'}
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_x=0.5
    )
    
    return fig

def create_amount_distribution_plot(df, predictions):
    """Create amount distribution plot"""
    plot_df = df.copy()
    plot_df['Transaction_Type'] = ['Anomaly' if x == 1 else 'Normal' for x in predictions]
    
    fig = px.histogram(
        plot_df,
        x='Amount',
        color='Transaction_Type',
        nbins=50,
        title="Transaction Amount Distribution",
        color_discrete_map={'Normal': '#4CAF50', 'Anomaly': '#F44336'}
    )
    
    fig.update_layout(height=400, title_x=0.5)
    return fig

def create_time_series_plot(df, predictions):
    """Create time series plot"""
    plot_df = df.copy()
    plot_df['Transaction_Type'] = ['Anomaly' if x == 1 else 'Normal' for x in predictions]
    plot_df['Hour'] = (plot_df['Time'] / 3600) % 24
    
    # Group by hour and transaction type
    hourly_counts = plot_df.groupby(['Hour', 'Transaction_Type']).size().reset_index(name='Count')
    
    fig = px.bar(
        hourly_counts,
        x='Hour',
        y='Count',
        color='Transaction_Type',
        title="Transactions by Hour of Day",
        color_discrete_map={'Normal': '#4CAF50', 'Anomaly': '#F44336'}
    )
    
    fig.update_layout(height=400, title_x=0.5)
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
    st.markdown("### Unsupervised Anomaly Detection System")
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    # Load models
    if st.sidebar.button("🚀 Load Models"):
        with st.spinner("Loading trained models..."):
            st.session_state.detector = load_detector()
            if st.session_state.detector:
                st.session_state.models_trained = True
                st.sidebar.success("✅ Models loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to load models")
    
    # Load sample data
    if st.sidebar.button("📊 Load Sample Data"):
        with st.spinner("Loading sample data..."):
            sample_data = load_sample_data()
            if sample_data is not None:
                st.session_state.sample_data = sample_data
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Sample data loaded!")
            else:
                st.sidebar.error("❌ Failed to load sample data")
    
    # File upload
    st.sidebar.markdown("### 📁 Upload Your Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with credit card transactions"
    )
    
    # Model selection
    if st.session_state.models_trained:
        st.sidebar.markdown("### 🤖 Model Selection")
        model_options = {
            'isolation_forest': 'Isolation Forest',
            'one_class_svm': 'One-Class SVM',
            'dbscan': 'DBSCAN'
        }
        selected_model = st.sidebar.selectbox(
            "Choose detection model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
    
    # Main content
    if not st.session_state.models_trained:
        st.info("👆 Please load the trained models using the sidebar to get started.")
        
        # Show project information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Project Overview")
            st.markdown("""
            This application demonstrates an **end-to-end unsupervised machine learning pipeline** 
            for detecting anomalies in credit card transactions. The system uses multiple algorithms:
            
            - **Isolation Forest**: Tree-based anomaly detection
            - **One-Class SVM**: Support vector machine for outlier detection  
            - **DBSCAN**: Density-based clustering for anomaly identification
            """)
        
        with col2:
            st.markdown("### 📈 Features")
            st.markdown("""
            - **Real-time Detection**: Upload CSV files for instant analysis
            - **Interactive Visualizations**: PCA plots, distributions, and time series
            - **Multiple Algorithms**: Compare different detection methods
            - **Performance Metrics**: Precision, recall, and F1-scores
            - **Deployment Ready**: Dockerized for easy deployment
            """)
        
        return
    
    # Data handling
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Uploaded file loaded! ({len(df)} transactions)")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")
    elif st.session_state.data_loaded:
        df = st.session_state.sample_data
    
    if df is None:
        st.info("📁 Please upload a CSV file or load sample data to begin analysis.")
        return
    
    # Data overview
    st.markdown("### 📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transactions", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Features", len(df.columns))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_amount = df['Amount'].mean() if 'Amount' in df.columns else 0
        st.metric("Avg Amount", f"${avg_amount:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Class' in df.columns:
            fraud_rate = (df['Class'].sum() / len(df)) * 100
            st.metric("Known Fraud Rate", f"{fraud_rate:.2f}%")
        else:
            st.metric("Known Fraud Rate", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Run detection
    if st.button("🔍 Run Anomaly Detection", type="primary"):
        with st.spinner(f"Running {model_options[selected_model]} detection..."):
            try:
                # Prepare data (remove Class column if present for prediction)
                prediction_data = df.drop('Class', axis=1) if 'Class' in df.columns else df
                
                # Get predictions
                predictions, scores = st.session_state.detector.predict_anomalies(
                    prediction_data, selected_model
                )
                
                # Get PCA components for visualization
                pca_components = st.session_state.detector.get_pca_components(prediction_data)
                
                # Store results in session state
                st.session_state.predictions = predictions
                st.session_state.scores = scores
                st.session_state.pca_components = pca_components
                st.session_state.current_model = selected_model
                
                st.success("✅ Detection completed!")
                
            except Exception as e:
                st.error(f"❌ Error during detection: {str(e)}")
                return
    
    # Display results if available
    if hasattr(st.session_state, 'predictions'):
        predictions = st.session_state.predictions
        scores = st.session_state.scores
        pca_components = st.session_state.pca_components
        
        # Results summary
        st.markdown("### 🎯 Detection Results")
        
        anomaly_count = np.sum(predictions)
        normal_count = len(predictions) - anomaly_count
        anomaly_rate = (anomaly_count / len(predictions)) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="normal-card">', unsafe_allow_html=True)
            st.metric("Normal Transactions", f"{normal_count:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="anomaly-card">', unsafe_allow_html=True)
            st.metric("Anomalous Transactions", f"{anomaly_count:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        # Transaction scatter plot
        fig1 = create_transaction_plot(df, predictions, scores, pca_components)
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount distribution
            fig2 = create_amount_distribution_plot(df, predictions)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Time series
            if 'Time' in df.columns:
                fig3 = create_time_series_plot(df, predictions)
                st.plotly_chart(fig3, use_container_width=True)
        
        # Detailed results table
        if st.checkbox("📋 Show Detailed Results"):
            result_df = df.copy()
            result_df['Predicted_Anomaly'] = predictions
            result_df['Anomaly_Score'] = scores
            result_df['Risk_Level'] = pd.cut(
                scores, 
                bins=[-np.inf, -0.5, 0, 0.5, np.inf], 
                labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk']
            )
            
            # Filter options
            st.markdown("#### Filter Results")
            col1, col2 = st.columns(2)
            
            with col1:
                show_anomalies = st.checkbox("Show only anomalies", value=False)
            
            with col2:
                risk_filter = st.multiselect(
                    "Filter by risk level:",
                    options=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'],
                    default=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk']
                )
            
            # Apply filters
            filtered_df = result_df.copy()
            if show_anomalies:
                filtered_df = filtered_df[filtered_df['Predicted_Anomaly'] == 1]
            if risk_filter:
                filtered_df = filtered_df[filtered_df['Risk_Level'].isin(risk_filter)]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download results
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"anomaly_detection_results_{st.session_state.current_model}.csv",
                mime="text/csv"
            )
        
        # Model evaluation (if true labels available)
        if 'Class' in df.columns:
            st.markdown("### 📊 Model Performance")
            
            try:
                evaluation = st.session_state.detector.evaluate_model(df, st.session_state.current_model)
                
                if evaluation:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        precision = evaluation['classification_report']['1']['precision']
                        st.metric("Precision (Fraud)", f"{precision:.4f}")
                    
                    with col2:
                        recall = evaluation['classification_report']['1']['recall']
                        st.metric("Recall (Fraud)", f"{recall:.4f}")
                    
                    with col3:
                        f1 = evaluation['classification_report']['1']['f1-score']
                        st.metric("F1-Score (Fraud)", f"{f1:.4f}")
                    
                    if evaluation['auc_score']:
                        st.metric("AUC Score", f"{evaluation['auc_score']:.4f}")
                    
                    # Confusion matrix
                    st.markdown("#### Confusion Matrix")
                    cm = evaluation['confusion_matrix']
                    
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Normal', 'Predicted Fraud'],
                        y=['Actual Normal', 'Actual Fraud'],
                        colorscale='Blues',
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 16}
                    ))
                    
                    fig_cm.update_layout(
                        title="Confusion Matrix",
                        title_x=0.5,
                        height=400
                    )
                    
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error evaluating model: {str(e)}")

if __name__ == "__main__":
    main()

