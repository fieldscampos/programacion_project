"""
Jose Manuel Alejandro Gonzalez Campos
Materia: Programación
Profesor: Juan Manuel Nava
Proyecto para exentar la materia de programación
Descripción: Este script realiza un análisis de datos de galaxias, incluyendo la carga de datos,
la limpieza de datos, el cálculo de estadísticas descriptivas, la creación de gráficos y la regresión lineal.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

"""Crear carpeta de salida (para organizar)"""

output_dir = "resultados"
os.makedirs(output_dir, exist_ok=True)

"""Cargar datos"""
df = pd.read_excel("galaxias_data (1).xlsx", usecols=[0, 1, 2, 3])
df.columns = ["raefcorkpg", "error_raefcorkpg", "muecorg", "error_muecorg"]
df = df.dropna()
df = df[df["raefcorkpg"] > 0]
df["log_raefcorkpg"] = np.log10(df["raefcorkpg"])

"""Medidas de tendencia central y dispersión"""
moda = stats.mode(df["log_raefcorkpg"], keepdims=True).mode[0]
media = np.mean(df["log_raefcorkpg"])
mediana = np.median(df["log_raefcorkpg"])
varianza = np.var(df["log_raefcorkpg"], ddof=1)
desviacion = np.std(df["log_raefcorkpg"], ddof=1)
covarianza = np.cov(df["log_raefcorkpg"], df["muecorg"])[0, 1]

"""Guardar resultados estadísticos"""
with open(os.path.join(output_dir, "estadisticas.txt"), "w") as f:
    f.write(f"Moda: {moda}\n")
    f.write(f"Media: {media}\n")
    f.write(f"Mediana: {mediana}\n")
    f.write(f"Varianza: {varianza}\n")
    f.write(f"Desviación estándar: {desviacion}\n")
    f.write(f"Covarianza: {covarianza}\n")

"""Regresión lineal"""
X = df[["log_raefcorkpg"]]
y = df["muecorg"]
modelo = LinearRegression()
modelo.fit(X, y)
pendiente = modelo.coef_[0]
intercepto = modelo.intercept_
r2 = r2_score(y, modelo.predict(X))

"""Guardar resultados de la regresión"""
with open(os.path.join(output_dir, "estadisticas.txt"), "a") as f:
    f.write(f"Ecuación de la regresión: y = {pendiente:.4f}x + {intercepto:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

"""Gráfico de dispersión y regresión lineal"""
df["prediccion"] = modelo.predict(X)
df[["log_raefcorkpg", "muecorg", "prediccion"]].to_csv(os.path.join(output_dir, "datos_modelo.csv"), index=False)

"""Gráfico de dispersión"""
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["log_raefcorkpg"], df["muecorg"], c=df["muecorg"], cmap="viridis", s=10, alpha=0.7)
plt.colorbar(sc, label="Muecorg")
plt.xlabel("Log(Raefcorkpg)")
plt.ylabel("Muecorg")
plt.title("Diagrama de Dispersión: Log(Raefcorkpg) vs Muecorg")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dispersogram.png"), dpi=300)
plt.savefig(os.path.join(output_dir, "dispersogram.pdf"), dpi=300)
plt.close()

"""Gráfico de regresion líneal"""
plt.figure(figsize=(8, 6))
sns.scatterplot(x="log_raefcorkpg", y="muecorg", hue="muecorg", palette="viridis", data=df, s=10, alpha=0.7)
plt.plot(df["log_raefcorkpg"], df["prediccion"], color='red', label="Regresión Lineal")
plt.xlabel("Log(Raefcorkpg)")
plt.ylabel("Muecorg")
plt.title("Regresión Lineal: Log(Raefcorkpg) vs Muecorg")
plt.text(0.05, 0.95, f"y = {pendiente:.2f}x + {intercepto:.2f}\n$R^2$ = {r2:.3f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="black"))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "regresion_lineal.png"), dpi=300)
plt.savefig(os.path.join(output_dir, "regresion_lineal.pdf"), dpi=300)
plt.close()
