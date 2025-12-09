import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model import predict_next_years  

st.set_page_config(page_title="Solar Pro Max - AI Powered", layout="wide", page_icon="solar_panel")

st.title("Solar Pro Max – AI-Powered Solar Planner")
st.markdown("### Real NASA Data + LSTM AI Forecast ")

# === INPUTS ===
col1, col2 = st.columns(2)
with col1:
    st.subheader("Location")
    lat = st.number_input("Latitude", value=19.0760, format="%.6f")
    lon = st.number_input("Longitude", value=72.8777, format="%.6f")

with col2:
    st.subheader("Market Settings")
    tariff = st.number_input("Current Electricity Rate (₹/kWh)", value=8.5, step=0.5)
    inflation = st.slider("Annual Tariff Increase (%)", 3.0, 12.0, 6.0) / 100
    lifespan = st.slider("Project Lifespan (Years)", 20, 30, 25)

st.sidebar.header("Compare 3 Solar Systems")
scenarios = ["Basic", "Premium", "Beast Mode"]
configs = {}

for name in scenarios:
    with st.sidebar.expander(f"{name} System", expanded=True):
        defaults = {
            "Basic": {"size": 6.0, "cost": 42000, "opex": 600},
            "Premium": {"size": 12.0, "cost": 40000, "opex": 800},
            "Beast Mode": {"size": 25.0, "cost": 38000, "opex": 1200}
        }[name]

        size = st.number_input(f"Size (kW)", 1.0, 100.0, defaults["size"], 0.5, key=f"size_{name}")
        cost_per_kw = st.number_input(f"Cost per kW (₹)", 30000, 60000, defaults["cost"], 1000, key=f"cost_{name}")
        opex_per_kw = st.number_input(f"O&M per kW/year (₹)", 300, 2000, defaults["opex"], key=f"om_{name}")

        configs[name] = {
            "size_kw": size,
            "capex_per_kw": cost_per_kw,
            "opex_per_kw": opex_per_kw,
            "color": {"Basic": "#1f77b4", "Premium": "#ff7f0e", "Beast Mode": "#2ca02c"}[name]
        }

# === RUN ANALYSIS ===
if st.button(f"Run AI-Powered {lifespan}-Year Solar Analysis", type="primary", use_container_width=True):
    with st.spinner("Fetching NASA data & running AI forecast (takes ~10–20 sec first time)..."):
        try:
            # This uses your model.py → returns dict with predicted
            result = predict_next_years(lat, lon,lifespan)

            # Extract AI-predicted monthly GHI for next  years
            predicted_years = result["predicted"]  # List of  lists (each 12 months)
            historical = np.array(result["historical"])

            # Use first predicted year as baseline (most accurate)
            baseline_monthly_ghi = np.array(predicted_years[0])  # Year 2025 on average each day in a month gets this much GHI
            days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
            annual_ghi = np.sum(baseline_monthly_ghi * days) # we are multiplying each month's avg GHI by number of days in that month and summing it up to get annual GHI
  # kWh/m²/year

            st.success("AI Forecast Loaded Successfully!")
            st.write(f"**AI Prediction**: Next {lifespan}-year average GHI: `{np.mean(predicted_years):.3f}` kWh/m²/day "
                     f"(Trend: **{result['trend_percent']:+.2f}%**)")

            # === Financial Calculation ===
            results = {}
            for name, cfg in configs.items():
                size = cfg["size_kw"]
                total_cost = size * cfg["capex_per_kw"]
                first_year_kwh = annual_ghi * size * 0.82  # System efficiency ~82%

                cumulative = -total_cost
                cashflows = [cumulative]

                for year in range(1, lifespan + 1):
                    prod = first_year_kwh * (1 - 0.005) ** (year - 1)  # 0.5% degradation
                    rate = tariff * (1 + inflation) ** (year - 1)
                    revenue = prod * rate
                    opex = size * cfg["opex_per_kw"]
                    net = revenue - opex
                    cumulative += net
                    cashflows.append(cumulative)

                payback = next((y for y, c in enumerate(cashflows) if c >= 0), None)
                if payback is None:
                    payback_str = f">{lifespan} years"
                else:
                    payback_str = f"{payback} years"


                results[name] = {
                    "size": size,
                    "cost": total_cost,
                    "first_year_kwh": int(first_year_kwh),
                    "payback": payback_str,
                    "profit": int(cumulative),
                    "cashflows": cashflows,
                    "color": cfg["color"]
                }

            # === DISPLAY RESULTS ===
            st.markdown("---")
            st.subheader("Your 3 Solar Investment Options")

            cols = st.columns(3)
            for idx, (name, r) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"### {name}")
                    st.metric("System Size", f"{r['size']:.1f} kW")
                    st.metric("Total Cost", f"₹{r['cost']:,.0f}")
                    st.metric("First Year Output", f"{r['first_year_kwh']:,} kWh")
                    st.metric("Payback Period", r['payback'])
                    st.metric(f"{lifespan}-Year Profit", f"₹{r['profit']:,.0f}")

                    if "years" in r["payback"] and r["payback"][0].isdigit():
                        years_val = int(r["payback"].split()[0])
                        if years_val <= 6:
                            st.success("MONEY PRINTER!")
                        elif years_val <= 9:
                            st.warning("Strong Investment")
                        else:
                            st.info("Long-term Savings")
                    else:
                        st.info("Long-term Savings")


            # === Cash Flow Chart ===
            st.subheader("Cumulative Profit Over Time")
            fig, ax = plt.subplots(figsize=(12, 6))
            for name, r in results.items():
                ax.plot(range(lifespan + 1), r["cashflows"], label=f"{name} ({r['size']:.0f}kW)", 
                        linewidth=3, color=r["color"])

            ax.axhline(0, color="black", linestyle="--", alpha=0.7)
            ax.set_xlabel("Years")
            ax.set_ylabel("Cumulative Profit (₹)")
            ax.set_title(f"{lifespan}-Year Solar Investment Cash Flow (AI Forecasted Sunlight)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # === Show AI Forecast Data ===
                        # === BEAUTIFUL HISTORICAL + AI FORECAST GRAPH (LIKE YOUR SCREENSHOT) ===
            with st.expander(f"AI Solar Radiation Forecast – 40+ Years History + Next {lifespan} Years", expanded=True):
                st.subheader("40+ Years of NASA Data + LSTM AI Prediction")

                # Extract data from model.py result
                historical_ghi = np.array(result["historical"])           # Full history
                predicted_ghi = np.array(result["predicted"]).flatten()  
                # Create the plot
                fig, ax = plt.subplots(figsize=(16, 8))

                # Historical data (blue)
                ax.plot(historical_ghi, label="NASA Historical Data (1984–2024)", 
                        color="#1f77b4", linewidth=2.2, alpha=0.9)

                # AI Forecast (red, dashed)
                future_x = np.arange(len(historical_ghi), len(historical_ghi) + len(predicted_ghi))
                ax.plot(future_x, predicted_ghi, label="AI Forecast (2025–" + str(2024 + lifespan) + ")", 
                        color="#d62728", linewidth=3, linestyle="--", alpha=0.95)

                # Styling — exactly like your original  
                ax.set_xlabel("Months", fontsize=14)
                ax.set_ylabel("Monthly Avg GHI (kWh/m²/day)", fontsize=14)
                ax.legend(fontsize=12, loc="upper left")
                ax.grid(True, alpha=0.3)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Check your model.py is in the same folder and working.")

