# solar_engine_pro.py  ← NO CLASSES • NASA POWER • FULLY FUNCTIONAL (Dec 2025)
import pvlib
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import requests
import os
import json
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Create cache folder
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True) # we store the data of nasa thingy here so that we dont need to call it again and again

geocoder = Nominatim(user_agent="solar-pro-simple") #object geocoder for converting name to coordinates of latitude and longitude

# ====================== GEOCODE ======================
def geocode_location(location: str):
    loc = geocoder.geocode(location) #converts location name to coordinates
    if not loc:
        raise ValueError("Location not found! Try a bigger city name.")
    return loc.latitude, loc.longitude, loc.address

# ====================== NASA POWER ======================
def get_nasa_power_data(lat: float, lon: float):
    cache_key = f"{lat:.4f}_{lon:.4f}"
    cache_file = os.path.join(CACHE_DIR, f"nasa_{hashlib.md5(cache_key.encode()).hexdigest()}.json")

    # Return cached data if exists (offline mode)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    # NASA API call (2023 = complete year, no 422 error)
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "format": "JSON",
        "start": "2023",
        "end": "2023",
        "user": os.getenv("NASA_POWER_EMAIL")
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        raw = r.json()

        # Handle both response formats
        if 'properties' in raw:
            ghi_dict = raw['properties'].get('parameter', {}).get('ALLSKY_SFC_SW_DWN', {}) #how much sun light hits the roof every day (months)
        else:
            ghi_dict = raw.get('parameter', {}).get('ALLSKY_SFC_SW_DWN', {}) 

        if not ghi_dict:
            raise ValueError("No solar data from NASA")

        # Extract 12 monthly values
        values = []
        for key in sorted(ghi_dict.keys()):
            val = float(ghi_dict[key])
            values.append(val if val != -999 else np.nan) #-999 means missing data, np.nan means not a number stores nan if val==-999

        values = values[:12]
        avg = np.nanmean(values) #calculates average of the amount of sunlight which comes in 12 months
        monthly = [v if not np.isnan(v) else avg for v in values] #if nan present, it stores avg for smooth calculations

        result = {
            "monthly_kwh_m2_day": monthly,
            "annual_kwh_m2": np.mean(monthly) * 365
        }

        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(result, f)

        return result

    except Exception as e:
        raise ValueError(f"NASA failed: {str(e)}. Check internet or email in .env")

# ====================== GENERATION ======================
def calculate_generation(lat, lon, system_kw, tilt=None, azimuth=180, efficiency=0.20):
    if system_kw <= 0:
        system_kw = 0.1  # Prevent zero division

    data = get_nasa_power_data(lat, lon)
    tilt = tilt or max(10, abs(lat))

    # Tilt & azimuth correction
    tilt_factor = 1.0 + 0.12 * np.sin(np.radians(90 - tilt)) * np.cos(np.radians(abs(azimuth - 180)))
    tilt_factor = max(0.88, min(1.12, tilt_factor))

    monthly_poa = [ghi * tilt_factor for ghi in data["monthly_kwh_m2_day"]]
    annual_kwh = sum(monthly_poa) * system_kw * efficiency * 0.85

    return {
        "annual_kwh": round(annual_kwh),
        "monthly_kwh": [round(m * 30.44 * system_kw * efficiency * 0.85) for m in monthly_poa],
        "peak_sun_hours": round(annual_kwh / system_kw, 2),
        "data_source": "NASA POWER (cached)" if os.path.exists(
            os.path.join(CACHE_DIR, f"nasa_{hashlib.md5(f'{lat:.4f}_{lon:.4f}'.encode()).hexdigest()}.json")
        ) else "NASA POWER (live)"
    }

# ====================== ECONOMICS ======================
def economic_analysis(annual_kwh, bill_monthly, cost_total, tariff_increase=0.06, degradation=0.005):
    if cost_total <= 0:
        return {"payback_years": "N/A", "total_savings_25y": 0, "roi_percent": 0}

    savings = []
    production = annual_kwh
    for year in range(1, 26):
        yearly_savings = min(production, bill_monthly * 12) * (1 + tariff_increase)**(year - 1)
        savings.append(yearly_savings)
        production *= (1 - degradation)

    total_savings = sum(savings)
    cumsum = np.cumsum(savings)
    payback = next((y for y, s in enumerate(cumsum, 1) if s >= cost_total), ">25")

    return {
        "payback_years": payback,
        "total_savings_25y": int(total_savings),
        "roi_percent": round((total_savings - cost_total) / cost_total * 100, 1)
    }
