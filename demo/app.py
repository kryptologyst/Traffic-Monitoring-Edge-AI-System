"""Streamlit demo application for traffic monitoring system."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.traffic_model import TrafficCongestionModel
from pipelines.data_pipeline import TrafficDataGenerator
from utils.logging_utils import setup_logging


# Page configuration
st.set_page_config(
    page_title="Traffic Monitoring Edge AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ DISCLAIMER</h4>
    <p><strong>THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. NOT FOR SAFETY-CRITICAL DEPLOYMENT.</strong></p>
    <p>This traffic monitoring system is designed for academic research, educational demonstrations, and proof-of-concept development. 
    It should NOT be used in production environments where traffic safety depends on its accuracy or reliability.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🚦 Traffic Monitoring Edge AI System</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")

# Model selection
model_path = st.sidebar.selectbox(
    "Select Model",
    options=["models/baseline_model.pth", "models/compressed/compressed_model.pth"],
    index=0,
)

# Simulation parameters
st.sidebar.subheader("Simulation Parameters")
n_samples = st.sidebar.slider("Number of Samples", 10, 1000, 100)
simulation_speed = st.sidebar.slider("Simulation Speed (samples/sec)", 0.1, 10.0, 1.0)
random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=1000)

# Display options
st.sidebar.subheader("Display Options")
show_raw_data = st.sidebar.checkbox("Show Raw Data", value=True)
show_predictions = st.sidebar.checkbox("Show Predictions", value=True)
show_metrics = st.sidebar.checkbox("Show Performance Metrics", value=True)


@st.cache_data
def load_model_and_config(model_path: str) -> tuple[Optional[TrafficCongestionModel], Optional[Dict[str, Any]]]:
    """Load model and configuration with caching."""
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Create model from config
            config = checkpoint['config']
            model = TrafficCongestionModel(**config['model'])
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model, config
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data
def generate_traffic_data(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate traffic data with caching."""
    generator = TrafficDataGenerator(seed=seed)
    features, labels = generator.generate_dataset(n_samples)
    return features, labels


def create_traffic_plot(features: np.ndarray, labels: np.ndarray, predictions: Optional[np.ndarray] = None) -> go.Figure:
    """Create interactive traffic monitoring plot."""
    df = pd.DataFrame(features, columns=["Vehicle Count", "Avg Speed", "Weather", "Hour"])
    df["Actual Status"] = ["Congested" if label == 1 else "Smooth" for label in labels]
    
    if predictions is not None:
        df["Predicted Status"] = ["Congested" if pred == 1 else "Smooth" for pred in predictions]
        df["Confidence"] = predictions
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add actual data points
    fig.add_trace(go.Scatter(
        x=df["Vehicle Count"],
        y=df["Avg Speed"],
        mode='markers',
        marker=dict(
            size=10,
            color=df["Actual Status"].map({"Smooth": "green", "Congested": "red"}),
            opacity=0.7,
        ),
        name="Actual Status",
        text=df["Actual Status"],
        hovertemplate="<b>%{text}</b><br>" +
                      "Vehicle Count: %{x}<br>" +
                      "Avg Speed: %{y} km/h<br>" +
                      "<extra></extra>",
    ))
    
    # Add predictions if available
    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=df["Vehicle Count"],
            y=df["Avg Speed"],
            mode='markers',
            marker=dict(
                size=8,
                color=df["Predicted Status"].map({"Smooth": "lightgreen", "Congested": "lightcoral"}),
                opacity=0.5,
                symbol="diamond",
            ),
            name="Predicted Status",
            text=df["Predicted Status"],
            hovertemplate="<b>Predicted: %{text}</b><br>" +
                          "Vehicle Count: %{x}<br>" +
                          "Avg Speed: %{y} km/h<br>" +
                          "<extra></extra>",
        ))
    
    fig.update_layout(
        title="Traffic Monitoring: Vehicle Count vs Speed",
        xaxis_title="Vehicle Count (per minute)",
        yaxis_title="Average Speed (km/h)",
        hovermode='closest',
        height=500,
    )
    
    return fig


def create_time_series_plot(features: np.ndarray, labels: np.ndarray, predictions: Optional[np.ndarray] = None) -> go.Figure:
    """Create time series plot of traffic data."""
    df = pd.DataFrame(features, columns=["Vehicle Count", "Avg Speed", "Weather", "Hour"])
    df["Actual Status"] = ["Congested" if label == 1 else "Smooth" for label in labels]
    df["Time"] = range(len(df))
    
    if predictions is not None:
        df["Predicted Status"] = ["Congested" if pred == 1 else "Smooth" for pred in predictions]
    
    fig = go.Figure()
    
    # Add vehicle count
    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["Vehicle Count"],
        mode='lines+markers',
        name="Vehicle Count",
        line=dict(color='blue'),
        yaxis='y',
    ))
    
    # Add speed
    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["Avg Speed"],
        mode='lines+markers',
        name="Avg Speed (km/h)",
        line=dict(color='orange'),
        yaxis='y2',
    ))
    
    # Add congestion status
    congestion_y = [1 if status == "Congested" else 0 for status in df["Actual Status"]]
    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=congestion_y,
        mode='markers',
        name="Congestion Status",
        marker=dict(
            color=df["Actual Status"].map({"Smooth": "green", "Congested": "red"}),
            size=8,
        ),
        yaxis='y3',
    ))
    
    fig.update_layout(
        title="Traffic Data Time Series",
        xaxis_title="Time (samples)",
        yaxis=dict(title="Vehicle Count", side="left"),
        yaxis2=dict(title="Speed (km/h)", side="right", overlaying="y"),
        yaxis3=dict(title="Congestion", side="right", overlaying="y", range=[-0.1, 1.1]),
        height=400,
    )
    
    return fig


def main():
    """Main application function."""
    
    # Load model
    model, config = load_model_and_config(model_path)
    
    if model is None:
        st.error("Model not found. Please ensure the model file exists.")
        st.stop()
    
    # Generate traffic data
    with st.spinner("Generating traffic data..."):
        features, labels = generate_traffic_data(n_samples, random_seed)
    
    # Run predictions
    predictions = None
    confidence_scores = None
    inference_times = []
    
    if show_predictions:
        with st.spinner("Running predictions..."):
            with torch.no_grad():
                for i, feature in enumerate(features):
                    input_tensor = torch.FloatTensor(feature).unsqueeze(0)
                    
                    start_time = time.time()
                    output = model(input_tensor)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    if i == 0:  # Initialize arrays on first iteration
                        predictions = []
                        confidence_scores = []
                    
                    prediction = int(torch.sigmoid(output) > 0.5)
                    confidence = float(torch.sigmoid(output))
                    
                    predictions.append(prediction)
                    confidence_scores.append(confidence)
            
            predictions = np.array(predictions)
            confidence_scores = np.array(confidence_scores)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Traffic Monitoring Dashboard")
        
        # Traffic visualization
        fig = create_traffic_plot(features, labels, predictions)
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series plot
        fig_ts = create_time_series_plot(features, labels, predictions)
        st.plotly_chart(fig_ts, use_container_width=True)
    
    with col2:
        st.subheader("System Status")
        
        # Model information
        st.markdown("### Model Information")
        st.info(f"**Model Type:** {config['model']['activation'].upper()} Neural Network")
        st.info(f"**Architecture:** {config['model']['hidden_dims']}")
        st.info(f"**Parameters:** {sum(p.numel() for p in model.parameters()):,}")
        
        # Performance metrics
        if show_metrics and predictions is not None:
            st.markdown("### Performance Metrics")
            
            # Calculate accuracy
            accuracy = np.mean(predictions == labels)
            st.metric("Accuracy", f"{accuracy:.3f}")
            
            # Calculate average confidence
            avg_confidence = np.mean(confidence_scores)
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            # Calculate average inference time
            avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
            st.metric("Avg Inference Time", f"{avg_inference_time:.2f} ms")
            
            # Calculate throughput
            throughput = 1.0 / np.mean(inference_times)
            st.metric("Throughput", f"{throughput:.1f} FPS")
        
        # Data summary
        st.markdown("### Data Summary")
        st.info(f"**Total Samples:** {len(features)}")
        st.info(f"**Congested:** {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
        st.info(f"**Smooth:** {len(labels) - np.sum(labels)} ({(1-np.mean(labels))*100:.1f}%)")
    
    # Raw data table
    if show_raw_data:
        st.subheader("Raw Data")
        
        df = pd.DataFrame(features, columns=["Vehicle Count", "Avg Speed", "Weather", "Hour"])
        df["Actual Status"] = ["Congested" if label == 1 else "Smooth" for label in labels]
        
        if predictions is not None:
            df["Predicted Status"] = ["Congested" if pred == 1 else "Smooth" for pred in predictions]
            df["Confidence"] = confidence_scores
        
        st.dataframe(df, use_container_width=True)
    
    # Real-time simulation
    st.subheader("Real-time Simulation")
    
    if st.button("Start Real-time Simulation", type="primary"):
        # Create placeholder for real-time updates
        placeholder = st.empty()
        
        # Initialize data generator
        generator = TrafficDataGenerator(seed=random_seed)
        
        # Simulation loop
        for i in range(20):  # Run for 20 samples
            # Generate new data
            new_features, new_labels = generator.generate_dataset(1)
            feature = new_features[0]
            label = new_labels[0]
            
            # Run prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(feature).unsqueeze(0)
                output = model(input_tensor)
                prediction = int(torch.sigmoid(output) > 0.5)
                confidence = float(torch.sigmoid(output))
            
            # Update display
            with placeholder.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Vehicle Count", f"{feature[0]:.0f}")
                    st.metric("Avg Speed", f"{feature[1]:.1f} km/h")
                
                with col2:
                    st.metric("Weather", "Rain" if feature[2] == 1 else "Clear")
                    st.metric("Hour", f"{feature[3]:.0f}")
                
                with col3:
                    actual_status = "Congested" if label == 1 else "Smooth"
                    predicted_status = "Congested" if prediction == 1 else "Smooth"
                    
                    st.metric("Actual Status", actual_status)
                    st.metric("Predicted Status", predicted_status)
                    st.metric("Confidence", f"{confidence:.3f}")
            
            # Wait for next sample
            time.sleep(1.0 / simulation_speed)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Traffic Monitoring Edge AI System | Research & Educational Use Only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
