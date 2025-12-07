import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import requests
import os

# ==================== LSTM MODEL ====================
class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Predict next value


# ==================== FETCH NASA DATA ====================
def fetch_nasa_monthly(lat, lon, start=1984, end=2024):
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON"
    }
    try:
        print(f"Fetching NASA monthly GHI data for ({lat}, {lon}) from {start} to {end}...")
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        ghi_dict = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

        values = []
        for key in sorted(ghi_dict.keys()):
            val = ghi_dict[key]
            if val not in (-999, None, -99):
                values.append(float(val))
            else:
                values.append(values[-1] if values else 5.0)
        print(f"Successfully fetched {len(values)} months of real NASA data")
        return np.array(values, dtype=np.float32)
    except Exception as e:
        print(f"NASA request failed: {e}")
        print("Using synthetic data for demo...")
        exit()


# ==================== MAIN PREDICTION FUNCTION ====================
def predict_next_25_years(lat, lon, model_path="solar_lstm_temp.pt"):
    # 1. Get historical data
    monthly_data = fetch_nasa_monthly(lat, lon) #here monthly_data is a numpy array
    if len(monthly_data) < 100:
        raise ValueError("Not enough data")

    # 2. Normalize
    mean_val = monthly_data.mean() # Calculate mean and std for normalization
    std_val = monthly_data.std()
    normalized = (monthly_data - mean_val) / std_val # Z-score normalization

    # 3. Create sequences
    seq_len = 24 # to predict next month based on past 24 months
    X, y = [], []
    for i in range(len(normalized) - seq_len):
        X.append(normalized[i:i + seq_len]) # sequences of 24 months
        y.append(normalized[i + seq_len])  # next month value
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # converts python list to torch tensor so that LSTM can process it properly and has numbers of format float32
    print(X)                                                            #and then we add an extra dimension at the end by unsqueeze(-1) to represent single feature input 
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    print(y)
    #we can think X as an excel sheet where each row is a sequence of 24 months and each column is a feature (here only 1 feature: GHI)

    # 4. Model
    model = SimpleLSTM(hidden_size=64, num_layers=2)
    if os.path.exists(model_path):
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print("Training LSTM model on historical data...")
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        for epoch in range(80):
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            if (epoch + 1) % 20 == 0: # Print every 20 epochs
                print(f"   Epoch {epoch+1:2d}/80 - Loss: {loss.item():.6f}")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved → {model_path}")

    # 5. Predict next 300 months == 25 yrs
    model.eval()
    with torch.no_grad():
        predictions = []
        current_seq = torch.tensor(normalized[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, 24, 1]
                          
        for _ in range(300):
            next_val = model(current_seq)              # [1, 1]
            predictions.append(next_val.item())
            # Slide window: remove oldest, append new prediction
            current_seq = torch.cat([current_seq[:, 1:, :], next_val.unsqueeze(-1)], dim=1)

    # 6. Denormalize
    predicted = np.array(predictions) * std_val + mean_val
    predicted = np.round(predicted, 3)

    # 7. Beautiful Plot
    plt.figure(figsize=(16, 8), dpi=120)
    x_hist = np.arange(len(monthly_data))
    x_future = np.arange(len(monthly_data), len(monthly_data) + 300)

    plt.plot(x_hist, monthly_data, label="NASA Historical Data (1984–2024)", color="#1f77b4", linewidth=2)
    plt.plot(x_future, predicted, label="AI Forecast (Next 25 Years)", color="#d62728", linewidth=2.8, linestyle="--")

    plt.axvline(len(monthly_data) - 1, color="gray", linestyle=":", linewidth=2, label="Forecast Begins")

    # Styling
    plt.title(f"Solar Radiation Forecast for Latitude {lat}°, Longitude {lon}°\n"
              "40+ Years NASA Data + LSTM AI Prediction", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Months", fontsize=14)
    plt.ylabel("Monthly Avg GHI (kWh/m²/day)", fontsize=14)
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True, alpha=0.3)

    # Stats box
    last_year = np.mean(monthly_data[-12:])
    forecast_avg = np.mean(predicted)
    change = (forecast_avg - last_year) / last_year * 100
    stats = f"Last Year Avg: {last_year:.3f}\nForecast Avg: {forecast_avg:.3f}\nChange: {change:+.2f}%"
    plt.text(0.02, 0.98, stats, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))

    plt.tight_layout()
    plt.show()

    # 8. Print results
    print("\n" + "="*60)
    print("AI FORECAST: NEXT 5 YEARS (Monthly GHI in kWh/m²/day)")
    print("="*60)
    years = [predicted[i:i+12] for i in range(0, 300, 12)]
    for i, year in enumerate(years):
        print(f"Year {2025 + i}: {year}")
    print(f"\nOverall Trend: {change:+.2f}% vs last year")

    return {
        "historical": monthly_data.tolist(),
        "predicted_25y": years,
        "trend_percent": round(change, 2)
    }


# ==================== RUN IT ====================
if __name__ == "__main__":
    # Change these coordinates to your location!
    LAT = 19.0760   # Example: Mumbai
    LON = 72.8777

    result = predict_next_25_years(LAT, LON)
