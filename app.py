# app.py
import streamlit as st
from streamlit_option_menu import option_menu
from main import SolarPlannerPro
from fpdf import FPDF
from PIL import Image
import base64
import os

st.set_page_config(page_title="Solar Pro Max", layout="wide", page_icon="⚡")

planner = SolarPlannerPro()

# Language Support (Add more later)
LANG = {
    "en": {"title": "Solar Planner Pro", "location": "Location", "analyze": "Analyze"},
    "hi": {"title": "सोलर प्लानर प्रो", "location": "स्थान", "analyze": "विश्लेषण करें"},
    "ar": {"title": "مخطط الطاقة الشمسية", "location": "الموقع", "analyze": "تحليل"}
}


st.title(f"⚡Solar Planner Tool")

# Sidebar - 3 Scenarios
st.sidebar.header("Compare 3 Systems")
scenarios = ["Basic", "Premium", "Max Power"]
configs = {}
for scenario in scenarios:
    with st.sidebar.expander(f"{scenario} System"):
        configs[scenario] = {
            "size": st.slider(f"Size (kW)", 1.0, 50.0, {"Basic":5.0,"Premium":10.0,"Max Power":20.0}[scenario], key=f"size_{scenario}"),
            "cost_per_kw": st.number_input("Cost/kW ($)", 800, 3000, {"Basic":1200,"Premium":1800,"Max Power":2500}[scenario], key=f"cost_{scenario}"),
            "panel_watt": st.selectbox("Panel Watt", [400,540,600], index={"Basic":0,"Premium":1,"Max Power":2}[scenario], key=f"watt_{scenario}"),
            "battery_kwh": st.slider("Battery (kWh)", 0.0, 50.0, {"Basic":0.0,"Premium":13.5,"Max Power":40.0}[scenario], key=f"bat_{scenario}")
        }

location = st.text_input("Enter City/Address", "Mumbai, India")
monthly_bill = st.number_input("Monthly Bill ($)", 50, 1000, 200)
roof_area = st.number_input("Roof Area (m²)", 20, 500, 120)
tilt = st.slider("Roof Tilt (°)", 0, 60, 20)
azimuth = st.slider("Roof Direction (0=North, 180=South)", 0, 360, 180)

if st.button("Run Full Analysis", type="primary"):
    try:
        lat, lon, addr = planner.geocode(location)
        st.success(f"Location: {addr.split(',')[0]} | {lat:.2f}°, {lon:.2f}°")

        results = {}
        for name, cfg in configs.items():
            max_kw = roof_area * 1000 / cfg["panel_watt"] * 0.85 / 1000
            actual_kw = min(cfg["size"], max_kw)
            gen = planner.calculate_generation(lat, lon, actual_kw, tilt, azimuth)
            eco = planner.economic_analysis(gen["annual_kwh"], monthly_bill, actual_kw * cfg["cost_per_kw"])
            results[name] = {**gen, **eco, "cost": actual_kw * cfg["cost_per_kw"], "max_possible_kw": max_kw}

        # Display Comparison
        col1, col2, col3 = st.columns(3)
        for i, (name, r) in enumerate(results.items()):
            with [col1, col2, col3][i]:
                st.subheader(f"{name}")
                st.metric("System Size", f"{r['annual_kwh']/1200:.1f} kW")
                st.metric("Annual Output", f"{r['annual_kwh']:,} kWh")
                st.metric("Payback", f"{r['payback_years']} yrs")
                st.metric("25Y Profit", f"${r['total_savings_25y']:,}")
                if r['total_savings_25y'] > 50000:
                    st.success("MONEY PRINTER")

        # PDF Export
        if st.button("Export PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Solar Pro Report", ln=1, align='C')
            if os.path.exists("assets/logo.png"):
                pdf.image("assets/logo.png", 10, 10, 33)
            pdf.ln(20)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Location: {addr}", ln=1)
            pdf.cell(0, 10, f"Best System: {max(results.keys(), key=lambda k: results[k]['total_savings_25y'])}", ln=1)
            pdf.output("Solar_Report.pdf")
            with open("Solar_Report.pdf", "rb") as f:
                st.download_button("Download PDF Report", f, "solar_report.pdf", "application/pdf")

    except Exception as e:
        st.error(f"Error: {e}")