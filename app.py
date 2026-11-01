from streamlit_js_eval import get_geolocation
import folium
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RoadSafe AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="auto"
)

DATA_FILE = Path("ETP_4_New_Data_Accidents.csv")
MODEL_FILE = Path("roadsafe_model.joblib")

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background: rgba(120, 120, 120, 0.08);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(120, 120, 120, 0.18);
}

h1, h2, h3 {
    letter-spacing: -0.4px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    # Generate date-based features
    if "Date" in df.columns:
        dates = pd.to_datetime(
            df["Date"],
            errors="coerce",
            dayfirst=True
        )

        df["Month"] = dates.dt.month
        df["Year"] = dates.dt.year

    # Generate hour from accident time
    if "Time_of_Accident" in df.columns:
        times = pd.to_datetime(
            df["Time_of_Accident"].astype(str),
            errors="coerce"
        )

        df["Hour"] = times.dt.hour

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    df = load_data()
except Exception as exc:
    st.error(f"DATASET ERROR: {type(exc).__name__}: {exc!r}")
    st.stop()

try:
    saved = load_model()
except Exception as exc:
    st.error(f"MODEL ERROR: {type(exc).__name__}: {exc!r}")
    st.stop()

# Support either a saved bundle or a bare sklearn pipeline.
if isinstance(saved, dict):
    model = saved.get("model", saved.get("pipeline"))
    features = saved.get("features")
else:
    model = saved
    features = None

if model is None:
    st.error("The model file does not contain a recognized model or pipeline.")
    st.stop()

if features is None:
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        st.error(
            "Cannot determine the trained feature list. "
            "Check the structure saved by train_model.py."
        )
        st.stop()

missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    st.error("Some model features are not present in the CSV.")

    st.write("Missing features:")
    st.write(missing_features)

    st.write("CSV columns:")
    st.write(df.columns.tolist())

    st.stop()

target = "Accident_Severity_C"

if target not in df.columns:
    st.error(f"Dataset is missing the target column: {target}")
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🚦 RoadSafe AI")
st.sidebar.caption("Intelligent Traffic Safety Platform")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Severity Prediction",
        "Accident Analytics",
        "GIS Hotspots",
        "Incident Reports",
        "AI Chatbot"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Historical dataset: ETP accident records")
st.sidebar.caption("Predictions are estimates, not guarantees.")

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":
    st.title("🚦 RoadSafe AI")
    st.subheader("Traffic Accident Analysis & Safety Intelligence")

    st.write(
        "Explore historical accident patterns, analyze road-safety "
        "conditions and access machine-learning severity predictions."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accident Records", f"{len(df):,}")
    c2.metric(
        "Severity Classes",
        str(df[target].nunique())
    )
    c3.metric(
        "Missing Values",
        f"{int(df.isna().sum().sum()):,}"
    )
    c4.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Severity Distribution")
        counts = df[target].value_counts().sort_index()
        st.bar_chart(counts)

    with right:
        st.subheader("Data Quality")
        quality = pd.DataFrame({
            "Missing values": df.isna().sum()
        }).sort_values("Missing values", ascending=False)

        st.dataframe(quality, use_container_width=True)

    st.info(
        "Severity labels are displayed using their dataset codes. "
        "Confirm their meanings from the official data dictionary."
    )

# --------------------------------------------------
# SEVERITY PREDICTION
# --------------------------------------------------

elif page == "Severity Prediction":
    st.title("🤖 Accident Severity Prediction")
    st.write(
        "Enter values for the model's trained features. "
        "Use the dataset's actual codes."
    )

    with st.form("prediction_form"):
        inputs = {}

        columns = st.columns(2)

        for index, feature in enumerate(features):
            with columns[index % 2]:
                if feature not in df.columns:
                    st.error(f"Missing feature: {feature}")
                    continue

                series = df[feature]

                if (
                    pd.api.types.is_numeric_dtype(series)
                    and series.nunique(dropna=True) > 20
                ):
                    median = series.median()
                    default = (
                        float(median)
                        if pd.notna(median)
                        else 0.0
                    )

                    inputs[feature] = st.number_input(
                        feature,
                        value=default
                    )
                else:
                    options = list(series.dropna().unique())

                    if series.isna().any():
                        options.append(None)

                    if not options:
                        st.warning(f"No available values for {feature}")
                        inputs[feature] = None
                    else:
                        selected = st.selectbox(
                            feature,
                            options=options,
                            format_func=lambda x: (
                                "Missing / Unknown"
                                if x is None
                                else str(x)
                            )
                        )
                        inputs[feature] = selected

        submitted = st.form_submit_button(
            "Predict Severity",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            input_df = pd.DataFrame([inputs])[features]

            prediction = model.predict(input_df)[0]

            st.success(
                f"Predicted severity class: {prediction}"
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_

                probability_df = pd.DataFrame({
                    "Severity class": classes,
                    "Estimated model probability": probabilities
                })

                st.subheader("Model Probability Distribution")

                st.bar_chart(
                    probability_df.set_index("Severity class")
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True
                )

                st.caption(
                    "These are model-estimated probabilities and "
                    "may not be calibrated real-world probabilities."
                )

            st.warning(
                "This prediction is an estimate. Do not use it as "
                "the sole basis for emergency or road-safety decisions."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

elif page == "Accident Analytics":
    st.title("📊 Accident Analytics")

    available_years = []

    if "Date" in df.columns:
        parsed_dates = pd.to_datetime(
            df["Date"],
            errors="coerce",
            dayfirst=True
        )

        years = parsed_dates.dt.year.dropna().astype(int).unique()
        available_years = sorted(years.tolist())

    filtered = df.copy()

    if available_years:
        selected_years = st.multiselect(
            "Filter by year",
            available_years,
            default=available_years
        )

        parsed_dates = pd.to_datetime(
            filtered["Date"],
            errors="coerce",
            dayfirst=True
        )

        filtered = filtered[
            parsed_dates.dt.year.isin(selected_years)
        ]

    st.metric("Records in Selection", f"{len(filtered):,}")

    st.subheader("Severity Distribution")
    st.bar_chart(
        filtered[target].value_counts().sort_index()
    )

    for column in [
        "Day_of_Week",
        "Road_Condition_F",
        "Weather_Conditions_H",
        "Road_Feature_E"
    ]:
        if column in filtered.columns:
            with st.expander(f"Analyze {column}"):
                st.bar_chart(
                    filtered[column].value_counts().sort_index()
                )

    st.caption(
        "Categorical values retain their original dataset codes "
        "until their official meanings are verified."
    )

# --------------------------------------------------
# GIS HOTSPOTS
# --------------------------------------------------

elif page == "GIS Hotspots":
    st.title("🗺️ RoadSafe AI — GIS Hotspots")

    st.info(
        "DEMO MAP: The colored grid cells below are illustrative. "
        "They do not represent verified accident hotspots."
    )

    st.write(
        "Explore the Nagpur road network. We will connect verified "
        "accident locations after establishing the correct highway "
        "and chainage-to-GPS mapping."
    )

    st.subheader("📍 Your Current Location")

    if st.button("Find My Location"):
        st.session_state["request_location"] = True

    if st.session_state.get("request_location", False):
        location = get_geolocation()

        if location:
            latitude = location["coords"]["latitude"]
            longitude = location["coords"]["longitude"]

            st.success(
                f"Location found: {latitude:.5f}, {longitude:.5f}"
            )

            # Save coordinates for map centering
            st.session_state["user_lat"] = latitude
            st.session_state["user_lon"] = longitude
        else:
            st.info(
                "Waiting for location permission. "
                "Allow location access in your browser."
            )

    # Demonstration map centered on Nagpur (or the user's location)
    user_lat = st.session_state.get("user_lat", 21.1458)
    user_lon = st.session_state.get("user_lon", 79.0882)

    m = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=14 if "user_lat" in st.session_state else 12,
        tiles="OpenStreetMap"
    )

    if "user_lat" in st.session_state:
        folium.Marker(
            [user_lat, user_lon],
            tooltip="Your current location",
            popup="You are here",
            icon=folium.Icon(color="blue", icon="user")
        ).add_to(m)

    # Add a current-location button using the browser's GPS.
    locate_control = """
    <div style="position: fixed; top: 80px; right: 15px; z-index: 9999;">
      <button onclick="locateUser()"
        style="padding: 10px 14px; border-radius: 8px;
               border: 1px solid #777; background: white;
               color: #222; font-weight: bold; cursor: pointer;">
        📍 My Location
      </button>
    </div>
    <script>
    function locateUser() {
      if (!navigator.geolocation) {
        alert("Location is not supported by this browser.");
        return;
      }

      navigator.geolocation.getCurrentPosition(
        function(position) {
          const lat = position.coords.latitude;
          const lon = position.coords.longitude;
          alert("Your location: " + lat.toFixed(5) + ", " + lon.toFixed(5));
        },
        function(error) {
          alert("Could not get your location. Allow location permission and try again.");
        }
      );
    }
    </script>
    """

    m.get_root().html.add_child(folium.Element(locate_control))

    # Illustrative grid cells only — not real accident hotspots.
    demo_grids = [
        {
            "name": "Demo Zone A",
            "center": [21.1458, 79.0882],
            "color": "red",
            "label": "Illustrative high concentration"
        },
        {
            "name": "Demo Zone B",
            "center": [21.1550, 79.1050],
            "color": "orange",
            "label": "Illustrative medium concentration"
        },
        {
            "name": "Demo Zone C",
            "center": [21.1300, 79.0750],
            "color": "yellow",
            "label": "Illustrative low concentration"
        }
    ]

    for zone in demo_grids:
        lat, lon = zone["center"]

        folium.Rectangle(
            bounds=[
                [lat - 0.004, lon - 0.004],
                [lat + 0.004, lon + 0.004]
            ],
            color=zone["color"],
            fill=True,
            fill_color=zone["color"],
            fill_opacity=0.35,
            tooltip=zone["name"],
            popup=zone["label"]
        ).add_to(m)

    st_folium(
        m,
        use_container_width=True,
        height=550,
        key="roadsafe_integrated_gis"
    )

    st.caption(
        "Prototype map only. Colored zones are illustrative, "
        "not verified accident locations or calculated risk levels."
    )

    # Show available accident location fields for analysis.
    possible_location_columns = [
        c for c in df.columns
        if "location" in c.lower() or "chainage" in c.lower()
    ]

    with st.expander("Inspect available accident location data"):
        if possible_location_columns:
            st.dataframe(
                df[possible_location_columns].head(20),
                use_container_width=True
            )
        else:
            st.warning("No location or chainage fields were found.")

# --------------------------------------------------
# INCIDENT REPORTS PLACEHOLDER
# --------------------------------------------------

elif page == "Incident Reports":
    st.title("🚨 Incident Reports")

    st.info(
        "Community reporting will be connected to a persistent "
        "database in a later phase. No report is submitted or "
        "stored by this placeholder page."
    )

    st.write(
        "The next implementation will include incident location, "
        "time observed, description and verification status."
    )

# --------------------------------------------------
# AI CHATBOT
# --------------------------------------------------

elif page == "AI Chatbot":
    st.title("🤖 RoadSafe AI Assistant")
    st.write(
        "Your assistant for accident severity, road safety, "
        "weather precautions, and historical accident data."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    def get_response(question):
        q = question.lower().strip()

        # Whole-word split so "hi" does not match words like "this".
        words = set(
            q.replace("?", " ").replace(",", " ")
             .replace(".", " ").replace("!", " ").split()
        )

        # Friendly greeting
        if (
            words & {"hello", "hi", "hey"}
            or any(p in q for p in [
                "good morning", "good afternoon", "good evening"
            ])
        ):
            return (
                "👋 Hello! I'm the RoadSafe AI Assistant. "
                "Ask me about road safety, accident severity, "
                "weather precautions, historical accident data, "
                "or emergency guidance. How can I help?"
            )

        # Emergency
        elif any(word in q for word in [
            "emergency", "injured", "accident happened",
            "witness", "after an accident"
        ]):
            return (
                "🚨 If someone is injured, move to safety if possible "
                "and call 112 in India for emergency assistance. "
                "Avoid putting yourself in danger. Do not move a "
                "seriously injured person unless there is immediate danger."
            )

        # Traffic
        elif any(word in q for word in [
            "traffic", "congestion", "traffic jam", "busy road"
        ]):
            return (
                "🚦 I don't have a live traffic feed connected, so I "
                "cannot verify current traffic conditions. Please use "
                "a live navigation service for current traffic information."
            )

        # Hotspots
        elif any(word in q for word in [
            "hotspot", "dangerous road", "accident prone"
        ]):
            return (
                "🗺️ Open GIS Hotspots to view the RoadSafe AI map. "
                "The demonstration zones are illustrative, not verified "
                "real-world accident hotspots."
            )

        # User location
        elif any(word in q for word in [
            "location", "where am i", "coordinates"
        ]):
            if (
                "user_lat" in st.session_state
                and "user_lon" in st.session_state
            ):
                return (
                    f"📍 Your detected coordinates are "
                    f"{st.session_state['user_lat']:.5f}, "
                    f"{st.session_state['user_lon']:.5f}. "
                    "These coordinates do not provide live traffic data."
                )

            return (
                "Open GIS Hotspots and click Find My Location first. "
                "Allow location access in your browser."
            )

        # Severity prediction
        elif any(word in q for word in [
            "severity", "serious accident", "fatal", "predict",
            "prediction", "random forest", "machine learning", "model"
        ]):
            return (
                "🤖 ACCIDENT SEVERITY PREDICTION\n\n"
                "RoadSafe AI uses a Random Forest Classifier trained on "
                "historical accident records to classify accident severity "
                "into four classes.\n\n"
                "The model considers recorded features such as accident "
                "time, location, road conditions, weather and vehicle types, "
                "along with other features in the training pipeline.\n\n"
                "Open Severity Prediction to try the model. Predictions are "
                "estimates, not guarantees, and should not be treated as "
                "proof that a road is safe or unsafe."
            )

        # Weather safety
        elif any(word in q for word in [
            "rain", "rainy", "fog", "weather",
            "wet road", "slippery", "storm"
        ]):
            return (
                "🌧️ SAFE DRIVING IN BAD WEATHER\n\n"
                "• Reduce speed and increase following distance.\n"
                "• Use appropriate lights when visibility is poor.\n"
                "• Avoid sudden braking and sharp turns.\n"
                "• Never drive through floodwater when its depth or road "
                "condition is unknown.\n"
                "• Stop somewhere safe if visibility becomes dangerously poor."
            )

        # Road safety
        elif any(word in q for word in [
            "road safety", "drive safely", "safe driving",
            "prevent accidents", "avoid accident", "helmet", "seat belt"
        ]):
            return (
                "🛣️ ROAD SAFETY TIPS\n\n"
                "• Wear a seat belt or a properly fastened helmet.\n"
                "• Follow speed limits and traffic signals.\n"
                "• Never use your phone while driving.\n"
                "• Maintain a safe following distance.\n"
                "• Never drive under the influence of alcohol or drugs."
            )

        # Dataset and analytics
        elif any(word in q for word in [
            "dataset", "data", "analytics", "how many records",
            "accident records", "records", "severity distribution"
        ]):
            return (
                "📊 ROADSAFE AI ANALYTICS\n\n"
                "The main historical accident dataset contains 8,116 "
                "records.\n\n"
                "Visit Accident Analytics to explore severity distribution "
                "and other historical patterns. These records describe past "
                "accidents and do not establish current traffic conditions."
            )

        # Explain the project
        elif any(word in q for word in [
            "what is roadsafe", "about this project",
            "what can you do", "help", "features"
        ]):
            return (
                "👋 Welcome to RoadSafe AI!\n\n"
                "I can help explain:\n\n"
                "• 🚦 Accident severity prediction\n"
                "• 📊 Historical accident analytics\n"
                "• 🗺️ GIS hotspot map limitations\n"
                "• 🌧️ Safe driving in poor weather\n"
                "• 🚨 What to do in an emergency\n"
                "• 📍 Your detected coordinates, if available\n\n"
                "Try asking: 'How does severity prediction work?'"
            )

        # Fallback
        else:
            return (
                "I'm the RoadSafe AI road-safety assistant. I couldn't "
                "confidently match that question to my supported topics.\n\n"
                "Try asking:\n"
                "• How does accident severity prediction work?\n"
                "• What should I do after witnessing an accident?\n"
                "• How can I drive safely in heavy rain?\n"
                "• Where can I view accident hotspots?\n"
                "• Is there live traffic near me?\n\n"
                "I cannot verify live traffic or actual accident hotspots "
                "without connected data sources."
            )

    # Display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user question
    question = st.chat_input("Ask RoadSafe AI a question...")

    if question:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": question
        })

        answer = get_response(question)

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()
