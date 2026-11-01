# 🚦 RoadSafe AI

### AI-Powered Road Accident Prediction & Hotspot Analysis

RoadSafe AI is an intelligent road safety platform designed to analyze accident data, predict accident severity, and identify potential accident hotspots using Machine Learning and GIS visualization.

Our goal is to support data-driven road safety planning and help identify areas that may require preventive attention.

## 🌟 Key Features

* **Accident Severity Prediction:** Predicts accident severity using a Machine Learning model.
* **Accident Analytics:** Visualizes accident patterns and trends.
* **GIS Hotspot Analysis:** Displays geographical accident insights.
* **Data Validation:** Analyzes and validates accident datasets.
* **Interactive Dashboard:** User-friendly interface built with Streamlit.

## 🧠 Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Random Forest Classifier
* Plotly
* GIS Visualization

## 📂 Project Structure

```text
RoadSafe-AI/
│
├── app.py
├── analyze_features.py
├── clean_chainage.py
├── evaluate_model.py
├── gis_map.py
├── inspect_data.py
├── train_model.py
├── validate_cleaned_data.py
│
├── Accdataset_cleaned_for_RoadSafeAI.csv
├── Accdataset_validated_for_RoadSafeAI.csv
└── ETP_4_New_Data_Accidents.csv
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/vedika-navghare/HACKOX1.git
```

Navigate to the project folder:

```bash
cd HACKOX1
```

Install the required libraries:

```bash
pip install streamlit pandas numpy scikit-learn plotly folium streamlit-folium joblib
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 🤖 Machine Learning

RoadSafe AI uses a Random Forest classification approach for accident severity prediction.

The model is trained using accident-related features and evaluated using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## 🗺️ GIS Analysis

The platform provides geographical visualization and accident hotspot analysis.

Accurate geographical positioning depends on the availability and validation of location-coordinate mapping.

## 🎯 Project Objective

To leverage Artificial Intelligence, Machine Learning, and GIS technologies to improve road safety analysis and support preventive decision-making.

## ⚠️ Disclaimer

This project is a hackathon prototype. Predictions and hotspot visualizations should be treated as analytical support, not as verified real-time accident warnings.

## 👨‍💻 Contributors

* Sarthak Gautre
* Vedika Navghare

---

**RoadSafe AI – Making Roads Safer Through Intelligence.** 🚦
