
import streamlit as st
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

# ----------------------
# AREA FUNCTIONS
# ----------------------

def calculate_polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2) - (y1 * x2)
    return abs(area) / 2

def gps_to_meters(points):
    ref_lat, ref_lon = points[0]
    meter_points = []
    for lat, lon in points:
        x = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
        y = (lat - ref_lat) * 111000
        meter_points.append((x, y))
    return meter_points

def calculate_area_from_gps(gps_points):
    meter_points = gps_to_meters(gps_points)
    return calculate_polygon_area(meter_points)

# ----------------------
# TRAIN PRICE MODEL
# ----------------------

np.random.seed(42)
data = []

for _ in range(50):
    area = np.random.randint(800, 3000)
    location = np.random.choice([2, 1, 0])  # Urban=2, Semi=1, Rural=0
    road = np.random.choice([0, 1])
    water = np.random.choice([0, 1])

    base_price = [1500, 3000, 5000][location]
    price = area * base_price
    price += road * 200000
    price += water * 100000
    price += np.random.randint(-50000, 50000)

    data.append([area, location, road, water, price])

df = pd.DataFrame(data, columns=["Area", "Location", "Road", "Water", "Price"])

X = df[["Area", "Location", "Road", "Water"]]
y = df["Price"]

price_model = LinearRegression()
price_model.fit(X, y)

# ----------------------
# TRAIN FRAUD MODEL
# ----------------------

fraud_data = []
for _ in range(100):
    area_dev = np.random.uniform(0, 30)
    boundary_shift = np.random.uniform(0, 10)
    price_dev = np.random.uniform(0, 25)
    fraud = 1 if (area_dev > 10 or boundary_shift > 5 or price_dev > 15) else 0
    fraud_data.append([area_dev, boundary_shift, price_dev, fraud])

fraud_df = pd.DataFrame(fraud_data,
                        columns=["Area_Dev", "Boundary_Shift", "Price_Dev", "Fraud"])

X_fraud = fraud_df[["Area_Dev", "Boundary_Shift", "Price_Dev"]]
y_fraud = fraud_df["Fraud"]

fraud_model = LogisticRegression()
fraud_model.fit(X_fraud, y_fraud)

# ----------------------
# STREAMLIT UI
# ----------------------

st.title("AI Smart Land Verification System")

st.header("Enter GPS Coordinates (4 points)")

lat1 = st.number_input("Latitude 1", value=11.0000)
lon1 = st.number_input("Longitude 1", value=76.0000)

lat2 = st.number_input("Latitude 2", value=11.0000)
lon2 = st.number_input("Longitude 2", value=76.0001)

lat3 = st.number_input("Latitude 3", value=11.0001)
lon3 = st.number_input("Longitude 3", value=76.0001)

lat4 = st.number_input("Latitude 4", value=11.0001)
lon4 = st.number_input("Longitude 4", value=76.0000)

recorded_area = st.number_input("Recorded Area (sq meters)", value=100.0)

location = st.selectbox("Location Type", ["Urban", "Semi", "Rural"])
road = st.selectbox("Road Access", [0, 1])
water = st.selectbox("Water Access", [0, 1])

if st.button("Analyze Land"):
    gps_points = [(lat1, lon1), (lat2, lon2), (lat3, lon3), (lat4, lon4)]
    calculated_area = calculate_area_from_gps(gps_points)

    location_map = {"Urban": 2, "Semi": 1, "Rural": 0}
    location_num = location_map[location]

    price_input = pd.DataFrame([[calculated_area, location_num, road, water]],
                               columns=["Area", "Location", "Road", "Water"])

    predicted_price = price_model.predict(price_input)[0]

    area_dev = abs(calculated_area - recorded_area) / recorded_area * 100

    fraud_input = pd.DataFrame([[area_dev, 3, 10]],
                               columns=["Area_Dev", "Boundary_Shift", "Price_Dev"])

    fraud_prob = fraud_model.predict_proba(fraud_input)[0][1]

    st.subheader("Results")

    st.write(f"📐 Calculated Area: {calculated_area:.2f} sq.m")
    st.write(f"💰 Predicted Price: ₹ {predicted_price:,.0f}")
    st.write(f"📊 Area Deviation: {area_dev:.2f} %")

    risk_level = "LOW"
    if fraud_prob > 0.7:
        risk_level = "HIGH"
    elif fraud_prob > 0.4:
        risk_level = "MEDIUM"

    st.write(f"🚨 Fraud Probability: {fraud_prob:.2f}")
    st.write(f"⚠️ Risk Level: {risk_level}")

    
