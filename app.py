# app.py
import streamlit as st
from dup import (
    calculate_generation,
    economic_analysis
)


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
        default_size = {"Basic": 5.0, "Premium": 10.0, "Max Power": 20.0}[scenario]  #the amount of power it can generate, larger system size, more electricity generated
        default_cost = {"Basic": 1200, "Premium": 1600, "Max Power": 2200}[scenario] # the price to install 1kw of power panel
        default_watt = {"Basic": 400, "Premium": 540, "Max Power": 600}[scenario] #the power produced by one single panel
        default_battery = {"Basic": 0.0, "Premium": 10.0, "Max Power": 30.0}[scenario] #the energy it can store

        configs[scenario] = {
            "size_kw": st.slider("System Size (kW)", 1.0, 50.0, default_size, key=f"size_{scenario}"),
            "cost_per_kw": st.number_input("Cost per kW (₹)", 800, 3000, default_cost, key=f"cost_{scenario}"),
            "panel_watt": st.selectbox("Panel Wattage", [400, 540, 600], index=[400,540,600].index(default_watt), key=f"watt_{scenario}"),
            "battery_kwh": st.slider("Battery Storage (kWh)", 0.0, 100.0, default_battery, step=5.0, key=f"bat_{scenario}")
        }

# Main inputs
col1, col2 = st.columns(2)
with col1:
    col1, col2 = st.columns(2)
    with col1:
        lat = st.text_input("Latitude", "19.0760", help="Copy from Google Maps")
    with col2:
        lon = st.text_input("Longitude", "72.8777", help="Copy from Google Maps")
    roof_area = st.number_input("Available Roof Area (m²)", 20, 1000, 150)
    lat=float(lat)
    lon = float(lon)
with col2:
    monthly_bill_inr = st.number_input("Average Monthly Electricity Bill (₹)", 1000, 50000, 8000) #electricity bill we get
    tilt = st.slider("Roof Tilt Angle (°)", 0, 60, 20) 
    azimuth = st.slider("Roof Direction (180° = South)", 0, 360, 180, help="180° is ideal in India")

# Run Analysis
if st.button("Run Full Analysis – Get Real NASA Results", type="primary", use_container_width=True):
    with st.spinner("Fetching NASA satellite data for your location..."): #buffer symbol
        try:
            results = {}

            for name, cfg in configs.items():
                # Calculate max possible kW from roof
                panel_area_m2 = 2.0  # ~2 m² per panel (approx)
                max_panels = int((roof_area * 0.85) / panel_area_m2) #how many panels we can fit
                max_kw_from_roof = max_panels * (cfg["panel_watt"] / 1000) #changing to kW
                actual_kw = min(cfg["size_kw"], max_kw_from_roof)
                actual_kw = round(actual_kw, 2)

                # Generation
                gen = calculate_generation( # gen is a dictionary of values
                    lat=lat,
                    lon=lon,
                    system_kw=actual_kw,
                    tilt=tilt,
                    azimuth=azimuth,
                    years_ahead=5
                )

                # Economics
                total_cost = actual_kw * cfg["cost_per_kw"]

                eco = economic_analysis(
                annual_kwh_first_year=gen["annual_kwh"],  # ← ONLY first year!
                monthly_bill=monthly_bill_inr,
                system_cost=total_cost
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

                    # FINAL BULLETPROOF VERSION
                    payback_num = r['payback_years']
                    if isinstance(payback_num, str):
                        badge_color = "gray"
                        badge_text = "Long-term Project"
                    elif payback_num <= 5:
                        badge_color = "green"
                        badge_text = "MONEY PRINTER"
                    elif payback_num <= 7:
                        badge_color = "green"
                        badge_text = "INSANE INVESTMENT"
                    elif payback_num <= 9:
                        badge_color = "orange"
                        badge_text = "Very Good Investment"
                    else:
                        badge_color = "gray"
                        badge_text = "Long-term Savings"

                    if badge_color == "green":
                        st.success(f"{badge_text}")
                    elif badge_color == "orange":
                        st.warning(f"{badge_text}")
                    else:
                        st.info(f"{badge_text}")
             

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Tip: Make sure NASA_POWER_EMAIL is set in Render environment variables.")

