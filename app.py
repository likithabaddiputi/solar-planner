# app.py
import streamlit as st
from streamlit_option_menu import option_menu
from main import (
    geocode_location,
    calculate_generation,
    economic_analysis
)
from fpdf import FPDF
import os
import base64

# Page config
st.set_page_config(page_title="Solar Pro Max", layout="wide", page_icon="⚡")

# Title
st.title("Solar Planner Tool ")

# Sidebar: 3 Scenario Comparison    
st.sidebar.header("Compare 3 Systems")
scenarios = ["Basic", "Premium", "Max Power"]
configs = {}

for scenario in scenarios:
    with st.sidebar.expander(f"{scenario} System", expanded=True):
        default_size = {"Basic": 5.0, "Premium": 10.0, "Max Power": 20.0}[scenario]
        default_cost = {"Basic": 1200, "Premium": 1600, "Max Power": 2200}[scenario]
        default_watt = {"Basic": 400, "Premium": 540, "Max Power": 600}[scenario]
        default_battery = {"Basic": 0.0, "Premium": 10.0, "Max Power": 30.0}[scenario]

        configs[scenario] = {
            "size_kw": st.slider("System Size (kW)", 1.0, 50.0, default_size, key=f"size_{scenario}"),
            "cost_per_kw": st.number_input("Cost per kW (₹)", 800, 3000, default_cost, key=f"cost_{scenario}"),
            "panel_watt": st.selectbox("Panel Wattage", [400, 540, 600], index=[400,540,600].index(default_watt), key=f"watt_{scenario}"),
            "battery_kwh": st.slider("Battery Storage (kWh)", 0.0, 100.0, default_battery, step=5.0, key=f"bat_{scenario}")
        }

# Main inputs
col1, col2 = st.columns(2)
with col1:
    location = st.text_input("Enter City/Location", "Mumbai, India")
    roof_area = st.number_input("Available Roof Area (m²)", 20, 1000, 150)
with col2:
    monthly_bill_inr = st.number_input("Average Monthly Electricity Bill (₹)", 1000, 50000, 8000)
    tilt = st.slider("Roof Tilt Angle (°)", 0, 60, 20)
    azimuth = st.slider("Roof Direction (180° = South)", 0, 360, 180, help="180° is ideal in India")

# Run Analysis
if st.button("Run Full Analysis – Get Real NASA Results", type="primary", use_container_width=True):
    with st.spinner("Fetching NASA satellite data for your location..."): #buffer symbol
        try:
            lat, lon, full_address = geocode_location(location)
            st.success(f"Found: {full_address.split(',')[0]} | {lat:.3f}°, {lon:.3f}°")

            results = {}

            for name, cfg in configs.items():
                # Calculate max possible kW from roof
                panel_area_m2 = 2.0  # ~2 m² per panel (approx)
                max_panels = int((roof_area * 0.85) / panel_area_m2)
                max_kw_from_roof = max_panels * (cfg["panel_watt"] / 1000)
                actual_kw = min(cfg["size_kw"], max_kw_from_roof)
                actual_kw = round(actual_kw, 2)

                # Generation
                gen = calculate_generation(
                    lat=lat,
                    lon=lon,
                    system_kw=actual_kw,
                    tilt=tilt,
                    azimuth=azimuth
                )

                # Economics
                total_cost = actual_kw * cfg["cost_per_kw"]
                eco = economic_analysis(
                    annual_kwh=gen["annual_kwh"],
                    bill_monthly=monthly_bill_inr,
                    cost_total=total_cost
                )

                results[name] = {
                    **gen,
                    **eco,
                    "actual_kw": actual_kw,
                    "total_cost": int(total_cost),
                    "max_possible_kw": round(max_kw_from_roof, 2)
                }

            # Display Results
            st.markdown("---")
            st.subheader("Your 3 Solar Options – Powered by NASA")

            cols = st.columns(3)
            for idx, (name, r) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"### {name}")
                    st.metric("System Size", f"{r['actual_kw']} kW")
                    st.metric("First Year Output", f"{r['annual_kwh']:,} kWh")
                    st.metric("Payback Period", f"{r['payback_years']} years")
                    st.metric("25-Year Profit", f"₹{r['total_savings_25y']:,}")

                    if r['payback_years'] != ">25" and int(r['payback_years']) <= 6:
                        st.success("MONEY PRINTER")
                    elif int(r['payback_years']) <= 8:
                        st.warning("Very Good Investment")
                    else:
                        st.info("Long-term Savings")

                    st.caption(f"Data: {r.get('data_source', 'NASA POWER')}")

            # PDF Export Button
            if st.button("Export Branded PDF Report", use_container_width=True):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 18)
                pdf.cell(0, 15, "Solar Pro Max – Your Custom Report", ln=1, align="C")

                if os.path.exists("assets/logo.png"):
                    pdf.image("assets/logo.png", x=10, y=8, w=40)

                pdf.ln(20)
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"Location: {full_address.split(',')[0]}", ln=1)
                pdf.cell(0, 10, f"Monthly Bill: ₹{monthly_bill_inr:,}", ln=1)
                pdf.cell(0, 10, f"Best Option: {max(results.keys(), key=lambda k: results[k]['total_savings_25y'])}", ln=1)
                pdf.cell(0, 10, f"25-Year Savings: Up to ₹{max(r['total_savings_25y'] for r in results.values()):,}", ln=1)

                pdf.output("solar_report.pdf")
                with open("solar_report.pdf", "rb") as f:
                    st.download_button(
                        "Download Your PDF Report",
                        data=f,
                        file_name="Solar_Pro_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Tip: Make sure NASA_POWER_EMAIL is set in Render environment variables.")

# Footer
st.markdown("---")
st.markdown("Built with NASA Satellite Data • Real-time Accuracy • Made for Indian Homes")
