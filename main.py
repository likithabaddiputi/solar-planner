# solar_engine_pro_fixed.py  ← NO CLASSES • NASA POWER (Hourly→Monthly) • FULLY FIXED
import pvlib
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import requests
import os
import json
import hashlib
from dotenv import load_dotenv
from datetime import datetime
from model import predict_future_from_nasa_data


load_dotenv()

# Create cache folder
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

geocoder = Nominatim(user_agent="solar-pro-simple")

# ====================== NASA POWER (Hourly -> Monthly converter) ======================
def get_nasa_power_data(lat: float, lon: float, year: int = 2025):
    """
    Fetch hourly ALLSKY_SFC_SW_DWN from NASA POWER, convert to monthly averages
    (kWh/m^2/day) suitable for the rest of the generator code.

    Returns:
        {
            "monthly_kwh_m2_day": [m1, m2, ..., m12],
            "annual_kwh_m2": float,
            "source": "cached" or "live",
            "raw_units": "<units string if present>"
        }
    """
    # cache key includes year and endpoint type
    cache_key = f"hourly_{year}_{lat:.4f}_{lon:.4f}"
    cache_file = os.path.join(CACHE_DIR, f"nasa_{hashlib.md5(cache_key.encode()).hexdigest()}.json")

    # Return cached data if exists (offline mode)
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
                cached["source"] = "NASA POWER (cached)"
                return cached
        except Exception:
            # if cache is corrupt, delete and continue to fetch live
            try:
                os.remove(cache_file)
            except Exception:
                pass

    # Build request for hourly API
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    start = f"20210101"
    end = f"20251202"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "format": "JSON",
        "start": start,     # YYYYMMDD
        "end": end,         # YYYYMMDD
        "header": True,
        "time-standard": "UTC"
        # intentionally no 'user' here; optional
    }

    try:
        r = requests.get(url, params=params, timeout=40)
        # raise_for_status will raise for 4xx/5xx
        r.raise_for_status()
        raw = r.json()
    except requests.exceptions.HTTPError as e:
        # Try to surface NASA's messages if present
        try:
            raw = r.json()
            print(raw)
            #seems like not so important just checks if NASA wants to send any messages
            msgs = raw.get("messages", []) #returns empty list if not present
            raise ValueError(f"NASA HTTP {r.status_code}: {msgs or str(e)}")
        except Exception:
            raise ValueError(f"NASA HTTP {getattr(r, 'status_code', '??')}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to contact NASA POWER: {str(e)}")

    # Validate response
    if raw is None:
        raise ValueError("NASA returned empty response")

    # If NASA sends error messages and no properties -> stop
    if "messages" in raw and raw.get("properties") is None:
        raise ValueError(f"NASA POWER error: {raw.get('messages')}")

    props = raw.get("properties")
    if props is None:
        # Some responses might place parameter in top-level 'parameter'
        props = {"parameter": raw.get("parameter")} if raw.get("parameter") else None
        if props is None:
            raise ValueError("NASA response missing 'properties' or 'parameter'")

    parameter = props.get("parameter")
    if parameter is None:
        raise ValueError("NASA response missing 'parameter' key inside 'properties'")

    ghi_dict = parameter.get("ALLSKY_SFC_SW_DWN") #monthly solar energy dict 
    if ghi_dict is None:
        raise ValueError("NASA response does not contain ALLSKY_SFC_SW_DWN data")

    # Try to detect units if provided in the 'parameters' section
    raw_units = ""
    parameters_meta = raw.get("parameters", {})
    if isinstance(parameters_meta, dict):
        meta = parameters_meta.get("ALLSKY_SFC_SW_DWN", {})
        raw_units = meta.get("units", "") or ""

    # Convert hourly series (timestamps as strings) to daily totals
    # GHI hourly values: often in W/m^2 (instantaneous) — convert to Wh/m^2 by summing hourly W/m^2 * 1 hour -> Wh/m^2
    # then to kWh/m^2 by dividing by 1000.
    # If units mention 'kwh' treat values as kWh (per hour) and sum directly.

    # Normalize ghi_dict: keys -> sorted list
    # ghi_dict example keys: "2023010100", "2023010101", ... (YYYYMMDDHH)
    try:
        items = sorted(ghi_dict.items()) #sorts keys and corresponding values
    except Exception:
        # If ghi_dict is not a dict, fail gracefully
        raise ValueError("Unexpected ALLSKY_SFC_SW_DWN structure from NASA")

    # Build daily sums
    daily_sums = {}        # date (YYYYMMDD) -> sum of hourly values (in kWh/m^2)
    daily_counts = {}      # number of valid hours available for that day
    for ts, v in items:
        # ts might be like '2023010100' or '20230101T00:00' or '2023-01-01T00:00'
        # Normalize timestamp string to YYYYMMDDHH
        if v is None:
            continue

        try:
            val = float(v)
        except Exception:
            continue

        # skip NASA fill value
        if val == -999: #-999 means missing data acc to NASA
            continue

        # normalize timestamp into YYYYMMDD and hour
        ts_str = str(ts)
        # handle different timestamp formats heuristically

        #This part gives us the date in YYYYMMDD format
        if len(ts_str) >= 10 and ts_str[4] == "-":  # ISO like 2023-01-01T00:00
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", ""))
                day = dt.strftime("%Y%m%d")
            except Exception:
                # fallback: take first 8 chars
                day = ts_str.replace("-", "")[:8]
        else:
            # simple case: YYYYMMDDHH or YYYYMMDDHHMM
            day = ts_str[:8]

        # Determine units & convert hourly value into kWh/m^2 for that hour
        # Heuristic rules:
        # - if units mention 'kwh' -> assume value is kWh/m^2 per hour (rare) -> keep as-is
        # - else assume W/m^2 instantaneous -> convert to kWh by val (W/m^2) * 1 hour / 1000 = kWh/m^2
        if raw_units and "kwh" in raw_units.lower():
            hour_kwh = val  # already kWh (per that hour)
        else:
            # assume W/m^2 instantaneous -> Wh/m^2 for 1 hour = W * 1h -> divide by 1000 -> kWh
            hour_kwh = val / 1000.0

        daily_sums[day] = daily_sums.get(day, 0.0) + hour_kwh #adds if existing else adds 0 to the value and adds to the dictionary
        daily_counts[day] = daily_counts.get(day, 0) + 1

    # Now compute daily kWh/m^2/day, handling missing hours:
    # If a day has many missing hours, we try to scale if enough hours exist; else mark NaN
    daily_values = {}
    for day, total_kwh in daily_sums.items():
        count = daily_counts.get(day, 0)
        if count == 0:
            daily_values[day] = np.nan
            continue
        # If some hours missing but most present, scale to full day
        # e.g., if count >= 18 (>=75% hours), scale total to 24-hour equivalent
        if count >= 18:
            scaled = total_kwh * (24.0 / count)
            daily_values[day] = scaled
        else:
            # Not enough hours to trust scaling: set as NaN (will be imputed later)
            daily_values[day] = np.nan

    # Create a DataFrame indexed by date to ease monthly aggregation
    # We want a row per day for the whole year (even days with no data)
    all_days = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
    df = pd.DataFrame(index=all_days)
    # map daily_values into df
    df["kwh_m2_day"] = [daily_values.get(d.strftime("%Y%m%d"), np.nan) for d in all_days]

    # Fill missing daily values using reasonable approach:
    # 1) If there are NaNs, replace them with the mean of available days (nanmean).
    # 2) This is conservative and keeps monthly averages smooth.
    if df["kwh_m2_day"].isna().all():
        # extreme: no valid data at all -> fallback to a default value (4.5 kWh/m2/day)
        df["kwh_m2_day"].fillna(4.5, inplace=True)
    else:
        daily_mean = float(np.nanmean(df["kwh_m2_day"]))
        df["kwh_m2_day"].fillna(daily_mean, inplace=True)

    # Now compute monthly average daily kWh/m2/day for each month (1..12)
    monthly = []
    for month in range(1, 13):
        month_days = df[df.index.month == month]["kwh_m2_day"].values
        if len(month_days) == 0:
            monthly.append(float(daily_mean))
        else:
            monthly.append(float(np.mean(month_days)))

    # Ensure we have 12 entries
    monthly = monthly[:12]

    result = {
        "monthly_kwh_m2_day": monthly,
        "annual_kwh_m2": float(np.mean(monthly) * 365.0),
        "raw_units": raw_units,
        "source": "NASA POWER (live)"
    }

    # Cache result for offline and fast reuse
    try:
        with open(cache_file, "w") as f:
            json.dump(result, f)
    except Exception:
        # cache errors shouldn't stop execution
        pass

    return result

# ====================== GENERATION ======================
def calculate_generation(lat, lon, system_kw, tilt=None, azimuth=180, efficiency=0.20, years_ahead=5):
    """
    Calculate current generation and project future annual generation based on predicted GHI trends.
    Returns:
        - current annual + monthly kWh
        - predicted annual kWh for next N years
        - peak sun hours
        - data source
    """
    if system_kw <= 0:
        system_kw = 0.1  # Prevent zero division

    # --- Get NASA data ---
    nasa_data = get_nasa_power_data(lat, lon)
    future = predict_future_from_nasa_data(nasa_data, years_ahead=years_ahead)

    tilt = tilt or max(10, abs(lat))
    # Simple tilt & azimuth factor
    tilt_factor = 1.0 + 0.12 * np.sin(np.radians(90 - tilt)) * np.cos(np.radians(abs(azimuth - 180)))
    tilt_factor = max(0.88, min(1.12, tilt_factor))

    # --- Current monthly/annual production ---
    monthly_poa = [ghi * tilt_factor for ghi in nasa_data["monthly_kwh_m2_day"]]
    annual_kwh = sum(monthly_poa) * system_kw * efficiency * 0.85
    monthly_kwh = [round(m * 30.44 * system_kw * efficiency * 0.85) for m in monthly_poa]

    # --- Future annual production ---
    avg_current_ghi = np.mean(nasa_data["monthly_kwh_m2_day"])
    future_annual_kwh = [
        round(annual_kwh * (ghi / avg_current_ghi)) for ghi in future["next_years"]
    ]

    return {
        "annual_kwh": round(annual_kwh),
        "monthly_kwh": monthly_kwh,
        "future_annual_kwh": future_annual_kwh,  # for future economics
        "peak_sun_hours": round(annual_kwh / system_kw, 2),
        "data_source": nasa_data.get("source", "NASA POWER (live)"),
        "future_message": future.get("message")
    }


# ====================== ECONOMICS ======================
def economic_analysis(annual_kwh, bill_monthly, cost_total, tariff_increase=0.06, degradation=0.005):
    """
    Compute payback, total savings, and ROI based on projected annual generation
    future_annual_kwh: list of kWh/year for each future year
    """
    if cost_total <= 0:
        return {"payback_years": "N/A", "total_savings_25y": 0, "roi_percent": 0}

    savings = []
    for year, production in enumerate(annual_kwh, start=1):
        # Apply degradation
        production_adj = production * ((1 - degradation) ** (year - 1))
        # Savings capped by actual bill
        yearly_savings = min(production_adj, bill_monthly * 12) * (1 + tariff_increase) ** (year - 1)
        savings.append(yearly_savings)

    total_savings = sum(savings)
    cumsum = np.cumsum(savings)
    payback = next((y for y, s in enumerate(cumsum, 1) if s >= cost_total), 26)

    return {
        "payback_years": payback,
        "total_savings_25y": int(total_savings),
        "roi_percent": round((total_savings - cost_total) / cost_total * 100, 1)
    }

get_nasa_power_data(19.0760,72.8777)
