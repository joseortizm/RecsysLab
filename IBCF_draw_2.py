import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

df = pd.read_csv("results_item_based_sensitivity_k.csv")

# Filtrar solo los experimentos con X_train_ratio = 0.8
df_filtered = df[df["X_train_ratio"] == 0.8].sort_values(by="k")

k_values = df_filtered["k"].values
mae_values = df_filtered["mae"].values

# Interpolación para curva suave
k_smooth = np.linspace(k_values.min(), k_values.max(), 300)
spline = make_interp_spline(k_values, mae_values, k=3)
mae_smooth = spline(k_smooth)

# Crear gráfico
plt.figure(figsize=(10, 5))
plt.plot(k_smooth, mae_smooth, label="Item-based CF", marker='s', color='blue')
plt.scatter(k_values, mae_values, color='blue', s=40)  # puntos reales

# Personalización
plt.title("Sensitivity of the Neighborhood Size (X = 0.8)")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
