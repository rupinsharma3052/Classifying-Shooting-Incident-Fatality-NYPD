import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.decomposition import PCA
import io
import warnings
warnings.filterwarnings("ignore")

# ── Matplotlib theme helper ───────────────────────────────────────────────────
def apply_mpl_theme():
    """Style all matplotlib figures for light or dark mode."""
    try:
        theme = st.get_option("theme.base")
    except Exception:
        theme = "light"
    is_dark = theme == "dark"
    bg   = "#0e1117" if is_dark else "#ffffff"
    fg   = "#fafafa" if is_dark else "#1a1a2e"
    grid = "#2a2a3a" if is_dark else "#e0e0e0"
    plt.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": bg,
        "axes.edgecolor": fg,   "axes.labelcolor": fg,
        "axes.titlecolor": fg,  "xtick.color": fg,
        "ytick.color": fg,      "text.color": fg,
        "grid.color": grid,     "legend.facecolor": bg,
        "legend.edgecolor": fg,
    })

CLR_A    = "#5b8dd9"   # blue  – "Not Murder"
CLR_B    = "#e07b54"   # amber – "Murder"
CLR_MAIN = "#5b8dd9"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYPD Shooting Incident Fatality Classifier",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark/light mode friendly) ────────────────────────────────────
st.markdown("""
<style>
    /* Uses Streamlit's built-in CSS variables so colours flip automatically
       between light and dark mode without any JavaScript. */

    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-color);          /* adapts to theme */
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.65;                     /* slightly muted but readable */
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color, rgba(128,128,128,0.2));
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: var(--text-color);
    }
    /* Round alert boxes */
    .stAlert { border-radius: 8px; }

    /* Sidebar background matches secondary surface in both modes */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }

    /* Tabs: active tab indicator uses the primary accent colour */
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: var(--primary-color) !important;
        color: var(--primary-color) !important;
    }

    /* Dataframe / table text stays readable in dark mode */
    .stDataFrame { color: var(--text-color); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔫 NYPD Shooting Incident Fatality Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict whether a shooting incident resulted in murder · Random Forest (Best Model)</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("**Best Model Identified:**")
    st.success("✅ Random Forest")

    st.markdown("---")
    st.markdown("**Features Used:**")
    st.markdown("""
- `X_COORD_CD` – NY State Plane X
- `Y_COORD_CD` – NY State Plane Y
- `Latitude`
- `Longitude`
""")

    st.markdown("---")
    st.markdown("**Target:**")
    st.markdown("`STATISTICAL_MURDER_FLAG`  \n`1 = Murder · 0 = Not Murder`")

    st.markdown("---")
    st.markdown("**Dataset:** NYPD Shooting Incident Historic Data")
    contamination = st.slider("Isolation Forest Contamination", 0.01, 0.15, 0.05, 0.01,
                              help="Fraction of data treated as outliers")
    test_size = st.slider("Test Split Size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random State", value=42, min_value=0, max_value=999)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Upload & EDA",
    "🔍 Outlier Analysis",
    "🤖 Model Training",
    "📈 Model Comparison",
    "🎯 Predict"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Data Upload & EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload `NYPD_Shooting_Incident_Data__Historic_.csv`",
        type=["csv"],
        help="Download from NYC Open Data"
    )

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.session_state["df_raw"] = df_raw

        st.success(f"✅ Loaded **{df_raw.shape[0]:,} rows** and **{df_raw.shape[1]} columns**.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{df_raw.shape[0]:,}")
        col2.metric("Total Columns", df_raw.shape[1])
        col3.metric("Missing Values", f"{df_raw.isnull().sum().sum():,}")

        with st.expander("🔎 Preview First 10 Rows"):
            st.dataframe(df_raw.head(10), use_container_width=True)

        with st.expander("📋 Column Info & Dtypes"):
            buf = io.StringIO()
            df_raw.info(buf=buf)
            st.text(buf.getvalue())

        with st.expander("📊 Descriptive Statistics"):
            st.dataframe(df_raw.describe(), use_container_width=True)

        with st.expander("❓ Missing Values Per Column"):
            miss = df_raw.isnull().sum().reset_index()
            miss.columns = ["Column", "Missing"]
            miss = miss[miss["Missing"] > 0].sort_values("Missing", ascending=False)
            if miss.empty:
                st.info("No missing values found.")
            else:
                st.dataframe(miss, use_container_width=True)

        # Target distribution
        st.subheader("🎯 Target Distribution")
        if "STATISTICAL_MURDER_FLAG" in df_raw.columns:
            target_counts = df_raw["STATISTICAL_MURDER_FLAG"].value_counts().reset_index()
            target_counts.columns = ["Murder Flag", "Count"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].bar(["Not Murder (False)", "Murder (True)"],
                        target_counts["Count"], color=["#4c72b0", "#dd8452"])
            axes[0].set_title("Shooting Fatality Distribution")
            axes[0].set_ylabel("Count")
            axes[1].pie(target_counts["Count"],
                        labels=["Not Murder", "Murder"],
                        autopct="%1.1f%%", colors=["#4c72b0", "#dd8452"])
            axes[1].set_title("Proportion")
            st.pyplot(fig)
            plt.close()

        # Histograms of numeric features
        st.subheader("📉 Histograms of Numerical Features")
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        selected_hist = st.multiselect("Select Columns", num_cols,
                                       default=["PRECINCT"] if "PRECINCT" in num_cols else num_cols[:1])
        if selected_hist:
            fig, axes = plt.subplots(1, len(selected_hist),
                                     figsize=(6 * len(selected_hist), 4))
            if len(selected_hist) == 1:
                axes = [axes]
            for ax, col in zip(axes, selected_hist):
                ax.hist(df_raw[col].dropna(), bins=50, color="#4c72b0", edgecolor="white")
                ax.set_title(col)
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("👆 Please upload the NYPD Shooting Incident CSV file to get started.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Outlier Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔍 Outlier Detection with Isolation Forest")

    if "df_raw" not in st.session_state:
        st.warning("⬅️ Please upload data in the **Data Upload & EDA** tab first.")
    else:
        df = st.session_state["df_raw"].copy()

        # Pre-process
        drop_cols = ['OCCUR_DATE', 'OCCUR_TIME', 'BORO', 'LOC_OF_OCCUR_DESC',
                     'LOC_CLASSFCTN_DESC', 'LOCATION_DESC', 'PERP_AGE_GROUP',
                     'PERP_SEX', 'PERP_RACE', 'VIC_AGE_GROUP', 'VIC_SEX',
                     'VIC_RACE', 'Lon_Lat', 'INCIDENT_KEY']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        df.dropna(subset=['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude',
                          'STATISTICAL_MURDER_FLAG'], inplace=True)

        le = LabelEncoder()
        df['STATISTICAL_MURDER_FLAG'] = le.fit_transform(df['STATISTICAL_MURDER_FLAG'])

        features = ['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude']
        X = df[features]
        y = df['STATISTICAL_MURDER_FLAG']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso = IsolationForest(contamination=contamination, random_state=int(random_state))
        outlier_labels = iso.fit_predict(X_scaled)
        n_outliers = (outlier_labels == -1).sum()
        n_inliers = (outlier_labels == 1).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", f"{len(outlier_labels):,}")
        c2.metric("Outliers Detected", f"{n_outliers:,}")
        c3.metric("Clean Samples", f"{n_inliers:,}")

        # PCA visualisation
        st.subheader("PCA Scatter – Before & After Outlier Removal")
        X_plot = X.copy()
        X_plot["outlier"] = outlier_labels
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_plot.drop(columns=["outlier"]))
        X_pca_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
        X_pca_df["outlier"] = outlier_labels

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["#dd8452" if o == -1 else "#4c72b0" for o in X_pca_df["outlier"]]
        axes[0].scatter(X_pca_df["PCA1"], X_pca_df["PCA2"], c=colors, alpha=0.4, s=5)
        axes[0].set_title("Before Outlier Removal (Blue=Inlier, Orange=Outlier)")
        axes[0].set_xlabel("PCA1"); axes[0].set_ylabel("PCA2")

        clean_df = X_pca_df[X_pca_df["outlier"] == 1]
        axes[1].scatter(clean_df["PCA1"], clean_df["PCA2"], color="#4c72b0", alpha=0.4, s=5)
        axes[1].set_title("After Outlier Removal")
        axes[1].set_xlabel("PCA1"); axes[1].set_ylabel("PCA2")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Boxplots
        st.subheader("Boxplots of Features")
        fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
        for ax, feat in zip(axes2, features):
            ax.boxplot(X[feat].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#4c72b0", color="#4c72b0"),
                       medianprops=dict(color="white"))
            ax.set_title(feat)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.heatmap(pd.DataFrame(X_scaled, columns=features).corr(),
                    annot=True, cmap="coolwarm", ax=ax3)
        ax3.set_title("Feature Correlation Heatmap")
        st.pyplot(fig3)
        plt.close()

        # Store cleaned data
        mask = outlier_labels == 1
        X_clean = X[mask]
        y_clean = y[mask]
        st.session_state["X_clean"] = X_clean
        st.session_state["y_clean"] = y_clean
        st.session_state["features"] = features
        st.success("✅ Clean dataset saved for model training.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Model Training
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 Train & Evaluate Models")

    if "X_clean" not in st.session_state:
        st.warning("⬅️ Please complete **Outlier Analysis** first.")
    else:
        X_clean = st.session_state["X_clean"]
        y_clean = st.session_state["y_clean"]

        models_available = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=int(random_state)),
            "Random Forest": RandomForestClassifier(random_state=int(random_state)),
            "Gradient Boosting": GradientBoostingClassifier(random_state=int(random_state)),
            "Support Vector Machine": SVC(),
            "Naive Bayes": GaussianNB(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
        }

        selected_models = st.multiselect(
            "Select Models to Train",
            list(models_available.keys()),
            default=["Logistic Regression", "Decision Tree", "Random Forest",
                     "Gradient Boosting", "Naive Bayes", "K-Nearest Neighbors"]
        )

        if st.button("🚀 Train Selected Models", type="primary"):
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=test_size, random_state=int(random_state)
            )

            results = []
            best_model_obj = None
            best_model_name = None
            best_f1 = 0

            progress = st.progress(0, text="Training models…")
            for i, name in enumerate(selected_models):
                model = models_available[name]
                pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                results.append({"Model": name, "Accuracy": acc,
                                 "Precision": prec, "Recall": rec, "F1 Score": f1})

                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = name
                    best_model_obj = pipeline
                    st.session_state["best_pipeline"] = pipeline
                    st.session_state["best_model_name"] = name
                    st.session_state["y_test"] = y_test
                    st.session_state["y_pred"] = y_pred
                    st.session_state["X_test"] = X_test

                progress.progress((i + 1) / len(selected_models),
                                   text=f"Trained {i+1}/{len(selected_models)}: {name}")

            progress.empty()
            results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
            st.session_state["results_df"] = results_df

            st.success(f"🏆 Best Model: **{best_model_name}** (F1 = {best_f1:.4f})")

            st.dataframe(
                results_df.style.highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1 Score"],
                    color="#d4edda"
                ).format("{:.4f}", subset=["Accuracy", "Precision", "Recall", "F1 Score"]),
                use_container_width=True
            )

            # Confusion matrix for best model
            st.subheader(f"Confusion Matrix – {best_model_name}")
            cm = confusion_matrix(y_test, st.session_state["y_pred"])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Not Murder", "Murder"],
                        yticklabels=["Not Murder", "Murder"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix – {best_model_name}")
            st.pyplot(fig)
            plt.close()

            # Classification report
            with st.expander("📋 Full Classification Report"):
                st.text(classification_report(y_test, st.session_state["y_pred"],
                                               target_names=["Not Murder", "Murder"]))
        elif "results_df" in st.session_state:
            st.info("ℹ️ Using previously trained results. Click **Train Selected Models** to retrain.")
            st.dataframe(st.session_state["results_df"], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Model Comparison Charts
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Model Performance Comparison")

    if "results_df" not in st.session_state:
        st.warning("⬅️ Train models in the **Model Training** tab first.")
    else:
        results_df = st.session_state["results_df"]

        # Bar chart per metric
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        palette = sns.color_palette("Blues_d", len(results_df))

        for ax, metric in zip(axes.flatten(), metrics):
            sorted_df = results_df.sort_values(metric, ascending=True)
            bars = ax.barh(sorted_df["Model"], sorted_df[metric], color=palette)
            ax.set_xlim(0, 1.05)
            ax.set_title(metric, fontsize=13, fontweight="bold")
            ax.set_xlabel("Score")
            for bar, val in zip(bars, sorted_df[metric]):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Radar / spider chart
        st.subheader("Radar Chart – All Models")
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        colors = sns.color_palette("tab10", len(results_df))

        for (_, row), color in zip(results_df.iterrows(), colors):
            values = [row[m] for m in metrics] + [row[metrics[0]]]
            ax2.plot(angles, values, "o-", linewidth=2, label=row["Model"], color=color)
            ax2.fill(angles, values, alpha=0.07, color=color)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics, fontsize=11)
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
        ax2.set_title("Model Comparison Radar", fontsize=13, fontweight="bold", pad=20)
        st.pyplot(fig2)
        plt.close()

        # Best model highlight
        best = results_df.iloc[0]
        st.success(f"🏆 **Best Model: {best['Model']}** | "
                   f"Acc: {best['Accuracy']:.4f} | "
                   f"F1: {best['F1 Score']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🎯 Predict Shooting Incident Fatality")

    if "best_pipeline" not in st.session_state:
        st.warning("⬅️ Train models in the **Model Training** tab first.")
    else:
        best_name = st.session_state.get("best_model_name", "Random Forest")
        st.info(f"Using **{best_name}** for prediction.")

        st.markdown("### Enter Incident Coordinates")
        col1, col2 = st.columns(2)
        with col1:
            x_coord = st.number_input("X_COORD_CD (NY State Plane X)",
                                      value=1000100.0, format="%.2f")
            latitude = st.number_input("Latitude", value=40.7128, format="%.6f",
                                       min_value=40.0, max_value=41.5)
        with col2:
            y_coord = st.number_input("Y_COORD_CD (NY State Plane Y)",
                                      value=237670.0, format="%.2f")
            longitude = st.number_input("Longitude", value=-74.0060, format="%.6f",
                                        min_value=-74.5, max_value=-73.5)

        st.markdown("---")
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            input_df = pd.DataFrame({
                "X_COORD_CD": [x_coord],
                "Y_COORD_CD": [y_coord],
                "Latitude": [latitude],
                "Longitude": [longitude],
            })

            pipeline = st.session_state["best_pipeline"]
            prediction = pipeline.predict(input_df)[0]

            # Probability if available
            proba_str = ""
            if hasattr(pipeline.named_steps["clf"], "predict_proba"):
                proba = pipeline.predict_proba(input_df)[0]
                proba_str = f"  \nConfidence: **{max(proba)*100:.1f}%**"

            if prediction == 1:
                st.error(f"⚠️ Prediction: **MURDER** (Fatal Incident){proba_str}")
            else:
                st.success(f"✅ Prediction: **NOT MURDER** (Non-Fatal Incident){proba_str}")

        st.markdown("---")
        st.markdown("### Batch Prediction")
        batch_file = st.file_uploader("Upload CSV with columns: X_COORD_CD, Y_COORD_CD, Latitude, Longitude",
                                      type=["csv"], key="batch")
        if batch_file:
            batch_df = pd.read_csv(batch_file)
            required = ["X_COORD_CD", "Y_COORD_CD", "Latitude", "Longitude"]
            if all(c in batch_df.columns for c in required):
                preds = st.session_state["best_pipeline"].predict(batch_df[required])
                batch_df["Prediction"] = ["Murder" if p == 1 else "Not Murder" for p in preds]
                st.dataframe(batch_df, use_container_width=True)
                csv_out = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Predictions", csv_out,
                                   "predictions.csv", "text/csv")
            else:
                st.error(f"CSV must contain columns: {required}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#888;font-size:0.85rem;'>"
    "NYPD Shooting Incident Fatality Classifier · Built with Streamlit · "
    "Best Model: Random Forest</p>",
    unsafe_allow_html=True
)
