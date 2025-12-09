import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import requests
import os

# ==================== LSTM MODEL ====================
class SimpleLSTM(nn.Module): #inheriting properties from nn.Module
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__() #super initializes the parent class nn.Module and helps to access its methods and properties
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1) #h1*w1+h2*w2+bias=output

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Predict next value something like [[1]]


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
        raise RuntimeError(f"NASA request failed: {e}")
        exit()


# ==================== MAIN PREDICTION FUNCTION ====================
def predict_next_years(lat, lon, lifespan, model_path="solar_lstm_temp.pt"):
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
      #X is a 3D tensor                                                        #and then we add an extra dimension at the end by unsqueeze(-1) to represent single feature input 
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
     #y is a 2D sensor
    #we can think X as an excel sheet where each row is a sequence of 24 months and each column is a feature (here only 1 feature: GHI)
    #LSTM works on 3D tensors of shape (batch_size, sequence_length, input_size)
    #unsquuese converts [[1,2],[3,4],[5,6]] to [[[1],[2]],[[3],[4]],[[5],[6]]]

    # 4. Model
    model = SimpleLSTM(hidden_size=64, num_layers=2)
    if os.path.exists(model_path):
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_path, map_location="cpu")) #loads the model parameters from the saved file into the model
    else:
        print("Training LSTM model on historical data...")
        model.train() #puts the model in training mode
        opt = torch.optim.Adam(model.parameters(), lr=0.001) #model.parameters() returns all the parameters of the model 
                                                             #that need to be optimized like weights and biases
        #the change in weights is gradient * learning_rate where gradient is calculated using backpropagation
        #using the formula loss/weight to get the gradient of loss with respect to that weight
        loss_fn = nn.MSELoss()
        for epoch in range(500):
            opt.zero_grad() #we are clearing the old gradients before calculating new gradients
             #so that gradients from previous epoch do not accumulate
            pred = model(X) #forward pass: passing input X through the model to get predictions
            loss = loss_fn(pred, y) #calculating loss between predicted and actual values using mean squared error
            loss.backward()  #computes the gradient of the loss with respect to model parameters using backpropagation formula loss/weight
            opt.step() #updates the model parameters using the calculated gradients and learning rate
            
            if (epoch + 1) % 20 == 0: # Print every 20 epochs
                print(f"   Epoch {epoch+1:2d}/80 - Loss: {loss.item():.6f}")
        torch.save(model.state_dict(), model_path) #saves the trained model parameters to a file
        print(f"Model saved → {model_path}")

    # 5. Predict next lifespan years
    model.eval() #puts the model in evaluation mode
    with torch.no_grad(): #since we are only doing inference and not training, we don't need to calculate gradients
        predictions = []
        current_seq = torch.tensor(normalized[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                          
        for _ in range(lifespan*12):  # Predict next 'lifespan' years
            next_val = model(current_seq)     
            predictions.append(next_val.item()) #.item() extracts the numerical value from a single-element tensor
            current_seq = torch.cat([current_seq[:, 1:, :], next_val.unsqueeze(-1)], dim=1)

    # 6. Denormalize
    predicted = np.array(predictions) * std_val + mean_val
    predicted = np.round(predicted, 3)

    last_year = np.mean(monthly_data[-12:])
    forecast_avg = np.mean(predicted)
    change = (forecast_avg - last_year) / last_year * 100

    print("\n" + "="*60)
    print(f"AI FORECAST: NEXT {lifespan} YEARS (Monthly GHI in kWh/m²/day)")
    print("="*60)
    years = [predicted[i:i+12] for i in range(0, lifespan*12, 12)]
    for i, year in enumerate(years):
        print(f"Year {2025 + i}: {year}")
    print(f"\nOverall Trend: {change:+.2f}% vs last year")

    return {
        "historical": monthly_data.tolist(),
        "predicted": years,
        "trend_percent": round(change, 2)
    }

# ==================== RUN IT ====================
if __name__ == "__main__":
    # Change these coordinates to your location!
    LAT = 19.0760   # Example: Mumbai
    LON = 72.8777

    result = predict_next_years(LAT, LON)
