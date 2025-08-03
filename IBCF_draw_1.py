import pandas as pd
import matplotlib.pyplot as plt

# Leer los resultados desde el archivo CSV
df = pd.read_csv("results_item_based_sensitivity_X.csv")

# Ordenar por X_train_ratio (eje horizontal)
df_sorted = df.sort_values(by="X_train_ratio")

# Crear el gráfico
plt.figure(figsize=(12, 5))
plt.plot(df_sorted["X_train_ratio"], df_sorted["mae"], marker='s', linestyle='-', color='red', label="item-item")

# Etiquetas y estilo
plt.title("Sensitivity of the parameter X (Train/Test Ratio)", fontsize=14)
plt.xlabel("Train/Test Ratio (X)", fontsize=12)
plt.ylabel("MAE", fontsize=12)
plt.xticks(df_sorted["X_train_ratio"], rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Guardar y mostrar

#plt.savefig("sensitivity_x_mae.png")
plt.show()





