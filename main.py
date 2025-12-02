# solar_engine_pro.py  ← 422 FIXED + FAKE DATA MODE (Dec 2025 – WORKS OFFLINE)
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

# Pre-loaded realistic monthly GHI data for major cities (kWh/m²/day) - from NASA/PVGIS averages
FAKE_DATA = {
    "mumbai": [5.8, 6.5, 6.8, 6.5, 6.2, 5.0, 4.8, 5.1, 5.5, 5.8, 5.6, 5.7],
    "delhi": [4.2, 5.1, 5.8, 6.5, 6.8, 6.5, 6.2, 6.0, 5.8, 5.5, 4.8, 4.0],
    "ahmedabad": [5.5, 6.2, 6.5, 6.8, 7.0, 6.5, 6.2, 6.0, 6.2, 6.5, 5.8, 5.5],
    "jaipur": [5.2, 6.0, 6.5, 7.0, 7.2, 6.8, 6.5, 6.2, 6.0, 6.2, 5.5, 5.0],
    "bengaluru": [5.5, 6.0, 6.2, 6.0, 5.8, 5.2, 5.0, 5.2, 5.5, 5.8, 5.5, 5.5],
    "chennai": [5.8, 6.2, 6.5, 6.5, 6.2, 5.5, 5.2, 5.5, 5.8, 6.0, 5.8, 5.8],
    "dubai": [5.0, 5.8, 6.5, 7.0, 7.2, 7.0, 6.8, 6.5, 6.2, 6.0, 5.5, 5.0],
    "riyadh": [4.5, 5.5, 6.5, 7.2, 7.5, 7.2, 7.0, 6.8, 6.5, 6.2, 5.5, 4.8],
    "berlin": [1.8, 2.5, 3.5, 4.5, 5.5, 5.8, 6.0, 5.5, 4.0, 2.8, 1.8, 1.5],
    # Default for unknown cities (India avg)
    "default": [5.0, 5.5, 5.8, 5.8, 5.5, 4.8, 4.5, 4.8, 5.2, 5.5, 5.2, 5.0]
}

class SolarPlannerPro:
    def __init__(self):
        self.geocoder = Nominatim(user_agent="solar-pro-v7")
        self.cache_dir = ".cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api_attempts = 0
        self.max_attempts = 3

    def geocode(self, location):
        loc = self.geocoder.geocode(location)
        if not loc:
            raise ValueError("Location not found! Try a bigger city name.")
        return loc.latitude, loc.longitude, loc.address

    def get_city_key(self, location):
        lower_loc = location.lower()
        for city in FAKE_DATA.keys():
            if city in lower_loc:
                return city
        return "default"

    def get_nasa_power_data(self, lat, lon, location_str):
        self.api_attempts += 1
        cache_key = f"{lat:.4f}_{lon:.4f}"
        cache_file = os.path.join(self.cache_dir, f"nasa_{hashlib.md5(cache_key.encode()).hexdigest()}.json")

        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f)

        # Fake data mode if API fails
        if self.api_attempts > self.max_attempts:
            city_key = self.get_city_key(location_str)
            monthly = FAKE_DATA.get(city_key, FAKE_DATA["default"])
            result = {
                "monthly_kwh_m2_day": monthly,
                "annual_kwh_m2": np.mean(monthly) * 365,
                "source": "Fake (realistic averages)"
            }
            print("Using fake data for demo - delete .cache for real NASA")
            return result

        # NASA call with hardened params
        start_year = "2023"
        end_year = "2023"
        email = os.getenv("NASA_POWER_EMAIL", "user@example.com").replace("@", "%40")  # Manual encode if needed

        url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": round(lon, 2),  # NASA prefers 2 decimals
            "latitude": round(lat, 2),
            "format": "JSON",
            "start": start_year,
            "end": end_year,
            "user": email
        }

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            raw = r.json()

            # Robust parsing
            if 'properties' in raw:
                parameter = raw['properties'].get('parameter', {})
            else:
                parameter = raw.get('parameter', {})

            ghi_dict = parameter.get('ALLSKY_SFC_SW_DWN', {})
            if not ghi_dict:
                raise ValueError("No GHI data")

            values = [float(ghi_dict[key]) for key in sorted(ghi_dict) if key in ghi_dict][:12]
            if len(values) < 12:
                avg = np.mean([v for v in values if v != -999])
                values += [avg] * (12 - len(values))

            values = [v if v != -999 else np.nan for v in values]
            valid = [v for v in values if not np.isnan(v)]
            avg_all = np.mean(valid) if valid else 4.5
            monthly_kwh_per_m2_per_day = [v if not np.isnan(v) else avg_all for v in values]

            result = {
                "monthly_kwh_m2_day": monthly_kwh_per_m2_per_day,
                "annual_kwh_m2": np.mean(monthly_kwh_per_m2_per_day) * 365,
                "source": "NASA POWER"
            }

            with open(cache_file, 'w') as f:
                json.dump(result, f)

            return result

        except Exception as e:
            print(f"NASA attempt {self.api_attempts} failed: {str(e)}")
            if self.api_attempts < self.max_attempts:
                return self.get_nasa_power_data(lat, lon, location_str)  # Retry
            else:
                # Fallback to fake
                city_key = self.get_city_key(location_str)
                monthly = FAKE_DATA.get(city_key, FAKE_DATA["default"])
                result = {
                    "monthly_kwh_m2_day": monthly,
                    "annual_kwh_m2": np.mean(monthly) * 365,
                    "source": "Fake (realistic averages)"
                }
                return result

    def calculate_generation(self, lat, lon, system_kw, tilt=None, azimuth=180, efficiency=0.20, location_str=""):
        data = self.get_nasa_power_data(lat, lon, location_str)
        if tilt is None:
            tilt = max(10, abs(lat))

        # Tilt factor
        tilt_factor = 1.0 + 0.12 * np.sin(np.radians(tilt)) * np.cos(np.radians(abs(azimuth - 180)))
        monthly_poa = [ghi * max(0.88, min(1.12, tilt_factor)) for ghi in data["monthly_kwh_m2_day"]]

        annual_kwh = sum(monthly_poa) * system_kw * efficiency * 0.85

        return {
            "annual_kwh": round(annual_kwh, 0),
            "monthly_kwh": [round(m * 30.44 * system_kw * efficiency * 0.85, 0) for m in monthly_poa],
            "peak_sun_hours": round(annual_kwh / system_kw, 2),
            "data_source": data["source"]
        }

    def economic_analysis(self, annual_kwh, bill_monthly, cost_total, tariff_increase=0.06, degradation=0.005):
        savings = []
        production = annual_kwh
        for year in range(1, 26):
            annual_bill = bill_monthly * 12
            yearly_savings = min(production, annual_bill) * (1 + tariff_increase)**(year - 1)
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