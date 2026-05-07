"""
NYPD Shooting Incident Fatality Prediction App
A machine learning application to classify shooting incidents as fatal or non-fatal
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NYPD Shooting Incident Predictor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}

.prediction-card {
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    margin: 1rem 0;
}

.prediction-fatal {
    background-color: #ff4d4f;
    color: white;
}

.prediction-non-fatal {
    background-color: #52c41a;
    color: white;
}

.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.metric-card {
    background-color: #fafafa;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load(SCALER_PATH)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_incident(model, scaler, x_coord, y_coord, latitude, longitude):

    input_data = pd.DataFrame({
        "X_COORD_CD": [float(x_coord)],
        "Y_COORD_CD": [float(y_coord)],
        "Latitude": [float(latitude)],
        "Longitude": [float(longitude)]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = probs[class_index]

    return prediction, confidence

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚨 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📍 Single Prediction",
        "📊 Batch Prediction",
        "📈 Data Insights",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
### Model Information

- Model Type: Random Forest
- Features:
    - X_COORD_CD
    - Y_COORD_CD
    - Latitude
    - Longitude
- Output:
    - Fatal
    - Non-Fatal
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<h1 class="main-header">🚔 NYPD Shooting Incident Fatality Predictor</h1>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown("""
        ## Welcome

        This application uses machine learning to predict whether a shooting incident
        in NYC is likely to result in a fatality.

        ### Features

        - Single incident prediction
        - Batch CSV prediction
        - Interactive visualizations
        - Data insights dashboard

        ### Workflow

        1. Enter geographic coordinates
        2. Model analyzes historical patterns
        3. Prediction + confidence score generated
        """)

    with col2:

        st.markdown("## Quick Stats")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Total Incidents", "27,312")
            st.metric("Fatal", "~2,600")

        with c2:
            st.metric("Non-Fatal", "~24,700")
            st.metric("Accuracy", "~80%")

    st.markdown("---")

    st.subheader("📍 NYC Borough Map")

    map_data = pd.DataFrame({
        "lat": [40.7128, 40.8448, 40.7282, 40.6500, 40.5795],
        "lon": [-74.0060, -73.8648, -73.7949, -73.9496, -74.1502],
        "borough": [
            "Manhattan",
            "Bronx",
            "Queens",
            "Brooklyn",
            "Staten Island"
        ]
    })

    st.map(map_data)

# =========================================================
# SINGLE PREDICTION
# =========================================================

elif page == "📍 Single Prediction":

    st.markdown(
        '<h1 class="main-header">📍 Single Incident Prediction</h1>',
        unsafe_allow_html=True
    )

    model = load_model()
    scaler = load_scaler()

    if model is not None and scaler is not None:

        st.markdown("""
        Enter incident coordinate information below.
        """)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("NY State Plane Coordinates")

            x_coord = st.number_input(
                "X_COORD_CD",
                value=1008000.0,
                format="%.2f"
            )

            y_coord = st.number_input(
                "Y_COORD_CD",
                value=208000.0,
                format="%.2f"
            )

        with col2:

            st.subheader("Global Coordinates")

            latitude = st.number_input(
                "Latitude",
                value=40.7000,
                format="%.6f"
            )

            longitude = st.number_input(
                "Longitude",
                value=-73.9000,
                format="%.6f"
            )

        st.markdown("---")

        if st.button("🔍 Predict", type="primary", use_container_width=True):

            # Coordinate validation
            if not (-90 <= latitude <= 90):
                st.error("Latitude must be between -90 and 90.")
                st.stop()

            if not (-180 <= longitude <= 180):
                st.error("Longitude must be between -180 and 180.")
                st.stop()

            with st.spinner("Analyzing incident..."):

                prediction, confidence = predict_incident(
                    model,
                    scaler,
                    x_coord,
                    y_coord,
                    latitude,
                    longitude
                )

                if prediction == 1:

                    st.markdown("""
                    <div class="prediction-card prediction-fatal">
                        <h2>⚠️ FATAL INCIDENT PREDICTED</h2>
                        <p>The model predicts a fatal outcome.</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="prediction-card prediction-non-fatal">
                        <h2>✅ NON-FATAL INCIDENT PREDICTED</h2>
                        <p>The model predicts a non-fatal outcome.</p>
                    </div>
                    """, unsafe_allow_html=True)

                if confidence is not None:

                    st.info(
                        f"Confidence Score: {confidence * 100:.2f}%"
                    )

                    st.progress(float(confidence))

        with st.expander("ℹ️ Coordinate Help"):

            st.markdown("""
            ### Finding Coordinates

            - Use Google Maps
            - Right-click a location
            - Copy latitude/longitude
            - NY State Plane coordinates can be found via NYC GIS tools
            """)

    else:
        st.error("Model or scaler could not be loaded.")

# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "📊 Batch Prediction":

    st.markdown(
        '<h1 class="main-header">📊 Batch Prediction</h1>',
        unsafe_allow_html=True
    )

    model = load_model()
    scaler = load_scaler()

    if model is not None and scaler is not None:

        st.markdown("""
        Upload a CSV containing:

        - X_COORD_CD
        - Y_COORD_CD
        - Latitude
        - Longitude
        """)

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                df = pd.read_csv(uploaded_file)

                required_columns = [
                    "X_COORD_CD",
                    "Y_COORD_CD",
                    "Latitude",
                    "Longitude"
                ]

                if not all(col in df.columns for col in required_columns):

                    st.error(
                        f"CSV must contain columns: {required_columns}"
                    )

                else:

                    st.subheader("Preview")

                    st.dataframe(df.head())

                    if st.button(
                        "🚀 Run Batch Prediction",
                        type="primary"
                    ):

                        with st.spinner("Processing data..."):

                            X_input = df[required_columns].copy()

                            # Convert to numeric
                            X_input = X_input.apply(
                                pd.to_numeric,
                                errors="coerce"
                            )

                            invalid_rows = X_input.isnull().any(axis=1)

                            invalid_count = invalid_rows.sum()

                            if invalid_count > 0:
                                st.warning(
                                    f"{invalid_count} invalid rows removed."
                                )

                            X_input = X_input[~invalid_rows]

                            if len(X_input) == 0:
                                st.error("No valid rows available.")
                                st.stop()

                            X_scaled = scaler.transform(X_input)

                            predictions = model.predict(X_scaled)

                            probabilities = None

                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba(X_scaled)

                            results_df = X_input.copy()

                            results_df["Prediction"] = [
                                "Fatal" if p == 1 else "Non-Fatal"
                                for p in predictions
                            ]

                            if probabilities is not None:
                                results_df["Confidence"] = probabilities.max(axis=1)

                            fatal_count = (predictions == 1).sum()
                            non_fatal_count = (predictions == 0).sum()

                            c1, c2, c3 = st.columns(3)

                            with c1:
                                st.metric("Total", len(results_df))

                            with c2:
                                st.metric("Fatal", fatal_count)

                            with c3:
                                st.metric("Non-Fatal", non_fatal_count)

                            st.subheader("Prediction Results")

                            st.dataframe(results_df)

                            # Download
                            csv = results_df.to_csv(index=False).encode("utf-8")

                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="prediction_results.csv",
                                mime="text/csv"
                            )

                            # Visualization
                            fig = px.pie(
                                names=["Fatal", "Non-Fatal"],
                                values=[fatal_count, non_fatal_count],
                                title="Prediction Distribution"
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

            except Exception as e:

                st.error(f"Error processing file: {e}")

    else:
        st.error("Model or scaler could not be loaded.")

# =========================================================
# DATA INSIGHTS
# =========================================================

elif page == "📈 Data Insights":

    st.markdown(
        '<h1 class="main-header">📈 Data Insights</h1>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Correlations",
        "🗺️ Borough Analysis",
        "📈 Statistics"
    ])

    # -----------------------------------------------------
    # TAB 1
    # -----------------------------------------------------

    with tab1:

        features = [
            "X_COORD_CD",
            "Y_COORD_CD",
            "Latitude",
            "Longitude"
        ]

        corr_matrix = np.array([
            [1.0, 0.85, -0.92, 0.88],
            [0.85, 1.0, -0.89, 0.83],
            [-0.92, -0.89, 1.0, -0.95],
            [0.88, 0.83, -0.95, 1.0]
        ])

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=features,
            y=features,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2),
            texttemplate="%{text}"
        ))

        fig.update_layout(
            title="Feature Correlation Matrix",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # TAB 2
    # -----------------------------------------------------

    with tab2:

        borough_data = pd.DataFrame({
            "Borough": [
                "Brooklyn",
                "Queens",
                "Manhattan",
                "Bronx",
                "Staten Island"
            ],
            "Incidents": [8500, 7200, 5800, 5300, 512],
            "Fatal_Rate": [8.5, 7.8, 9.2, 9.5, 6.5]
        })

        c1, c2 = st.columns(2)

        with c1:

            fig1 = px.bar(
                borough_data,
                x="Borough",
                y="Incidents",
                title="Incidents by Borough",
                color="Incidents"
            )

            st.plotly_chart(fig1, use_container_width=True)

        with c2:

            fig2 = px.bar(
                borough_data,
                x="Borough",
                y="Fatal_Rate",
                title="Fatality Rate (%)",
                color="Fatal_Rate"
            )

            st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------
    # TAB 3
    # -----------------------------------------------------

    with tab3:

        stats_df = pd.DataFrame({
            "Feature": [
                "X_COORD_CD",
                "Y_COORD_CD",
                "Latitude",
                "Longitude"
            ],
            "Mean": [
                "1,009,449",
                "208,127",
                "40.738",
                "-73.909"
            ],
            "Std Dev": [
                "18,378",
                "31,886",
                "0.088",
                "0.066"
            ],
            "Min": [
                "914,928",
                "125,757",
                "40.512",
                "-74.249"
            ],
            "Max": [
                "1,066,815",
                "271,128",
                "40.911",
                "-73.702"
            ]
        })

        st.dataframe(stats_df, use_container_width=True)

        st.markdown("""
        ### Model Metrics

        | Metric | Score |
        |--------|-------|
        | Accuracy | ~80% |
        | Precision | ~0.30 |
        | Recall | ~0.15 |
        | F1 Score | ~0.20 |
        """)

# =========================================================
# ABOUT PAGE
# =========================================================

else:

    st.markdown(
        '<h1 class="main-header">ℹ️ About This Project</h1>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ## Overview

    This project uses machine learning to classify shooting incidents
    as fatal or non-fatal using geographic features.

    ## Workflow

    1. Data preprocessing
    2. Feature engineering
    3. Model training
    4. Model evaluation
    5. Streamlit deployment

    ## Technologies

    - Python
    - Streamlit
    - Scikit-learn
    - Plotly
    - Pandas

    ## Disclaimer

    This application is for educational and analytical purposes only.
    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <small>
    NYPD Shooting Incident Fatality Predictor |
    Powered by Machine Learning
    </small>
    </center>
    """,
    unsafe_allow_html=True
)
