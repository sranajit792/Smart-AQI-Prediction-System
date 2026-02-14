import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart AQI Dashboard", layout="wide")

# ---------------- LOAD MODEL & DATA ----------------
model = joblib.load("model.pkl")
df = pd.read_csv("data.csv")
df = df.dropna(subset=["AQI"])
df["Date"] = pd.to_datetime(df["Date"])

features = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
X = df[features].fillna(df[features].mean())
y = df["AQI"]
preds = model.predict(X)
r2 = r2_score(y, preds)
mae = mean_absolute_error(y, preds)

# ---------------- LOCATION ----------------
def get_location():
    try:
        res = requests.get("https://ipinfo.io/json").json()
        city = res.get("city", "")
        country = res.get("country", "")
        loc = res.get("loc", "0,0")
        lat, lon = loc.split(",")
        return city, country, float(lat), float(lon)
    except:
        return "Unknown", "Unknown", 20.5937, 78.9629

city_name, country_name, lat, lon = get_location()
location = f"{city_name}, {country_name}"

# ---------------- LIGHT BLUE BACKGROUND & STYLING ----------------
st.markdown("""
<style>
.stApp {
    background-color: #a0d8f1;
    color: black;
    font-family: 'Segoe UI', sans-serif;
}

/* Small centered AQI image (logo style) */
.aqi-logo {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 120px;
    height: 120px;
    object-fit: cover;
    border-radius: 10px;
}

/* Glass panel styling */
.glass {
    background: rgba(255, 255, 255, 0.7);
    padding: 15px;
    border-radius: 15px;
    backdrop-filter: blur(8px);
    box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    text-align: center;
    margin-bottom: 20px;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 16px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TOP AQI IMAGE & TITLE ----------------
st.image("aqi.jpeg", width=120)  # Small square AQI image

st.markdown(f"""
<div class="glass">
<h1 style='font-size:32px; font-weight:700;'>🌍 SMART AI AIR QUALITY DASHBOARD</h1>
<h3>📍 {location}</h3>
<h3>⏰ {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h3>
</div>
""", unsafe_allow_html=True)

# ---------------- AQI CATEGORY ----------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00e400", "Air quality is satisfactory."
    elif aqi <= 100:
        return "Moderate", "#ffff00", "Sensitive individuals reduce outdoor activity."
    elif aqi <= 200:
        return "Poor", "#ff7e00", "Wear mask. Avoid prolonged exposure."
    elif aqi <= 300:
        return "Very Poor", "#ff0000", "Wear N95 mask."
    else:
        return "Severe", "#8f3f97", "Stay indoors. Use air purifier."

cities = sorted(df["City"].unique())
selected_city = st.selectbox("Select City", cities)
city_data = df[df["City"] == selected_city].sort_values("Date")
latest = city_data.iloc[-1]
category, color, advice = aqi_category(latest["AQI"])

st.markdown(f"""
<div class="glass">
<h2>Latest AQI: {latest["AQI"]}</h2>
<h3 style="color:{color}">{category}</h3>
<p>{advice}</p>
</div>
""", unsafe_allow_html=True)

# Sticker if AQI > 50
if latest["AQI"] > 50:
    st.image("sticker.png", width=100)

# ---------------- COLLAPSIBLE SECTIONS ----------------
with st.expander("📊 AQI Speedometer"):
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest["AQI"],
        gauge={'axis': {'range': [0, 500]}}
    ))
    gauge.update_layout(template="plotly_dark")
    st.plotly_chart(gauge, use_container_width=True)

with st.expander("🗺️ Live AQI Map"):
    map_df = pd.DataFrame({"lat":[lat],"lon":[lon],"AQI":[latest["AQI"]]})
    map_fig = px.scatter_mapbox(
        map_df, lat="lat", lon="lon",
        size="AQI", color="AQI",
        zoom=8, height=400
    )
    map_fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(map_fig, use_container_width=True)

with st.expander("📈 Historical AQI Trend"):
    fig = px.line(city_data, x="Date", y="AQI", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🧭 Pollutant Radar Chart"):
    values = latest[features].values.tolist()
    values += values[:1]
    radar_labels = features + [features[0]]
    fig_radar = go.Figure(data=[go.Scatterpolar(r=values, theta=radar_labels, fill='toself')])
    fig_radar.update_layout(template="plotly_dark")
    st.plotly_chart(fig_radar, use_container_width=True)

with st.expander("🤖 Predict AQI"):
    pm25 = st.number_input("PM2.5")
    pm10 = st.number_input("PM10")
    no2 = st.number_input("NO2")
    co = st.number_input("CO")
    so2 = st.number_input("SO2")
    o3 = st.number_input("O3")

    if st.button("Predict AQI"):
        input_data = np.array([[pm25, pm10, no2, co, so2, o3]])
        prediction = model.predict(input_data)[0]
        category, color, advice = aqi_category(prediction)
        st.markdown(f"""
        <div class="glass">
        <h2>{prediction:.2f}</h2>
        <h3 style="color:{color}">{category}</h3>
        <p>{advice}</p>
        </div>
        """, unsafe_allow_html=True)
        if prediction > 50:
            st.image("sticker.png", width=100)

with st.expander("📄 Model Performance"):
    st.write(f"R² Score: {r2:.3f}")
    st.write(f"MAE: {mae:.2f}")

with st.expander("📃 Download AQI Report"):
    def generate_pdf():
        file_path = "AQI_Report.pdf"
        doc = SimpleDocTemplate(file_path)
        elements = []
        styles = getSampleStyleSheet()
        elements.append(Paragraph("AQI Report", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"City: {selected_city}", styles["Normal"]))
        elements.append(Paragraph(f"AQI: {latest['AQI']}", styles["Normal"]))
        elements.append(Paragraph(f"Category: {category}", styles["Normal"]))
        elements.append(Paragraph(f"Advice: {advice}", styles["Normal"]))
        doc.build(elements)
        return file_path

    if st.button("Download AQI Report"):
        pdf_path = generate_pdf()
        with open(pdf_path, "rb") as f:
            st.download_button("Click to Download", f, file_name="AQI_Report.pdf")
