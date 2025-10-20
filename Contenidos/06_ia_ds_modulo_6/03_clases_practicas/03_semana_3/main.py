import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import time
import threading

pio.renderers.default = "browser"

def ejecutar_con_timeout(func, timeout=10, nombre="grafico"):
    resultado = None
    error = None
    
    def target():
        nonlocal resultado, error
        try:
            resultado = func()
        except Exception as e:
            error = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join()
    
    if thread.is_alive():
        print(f"Timeout en {nombre} - Continuando...")
        return False
    elif error:
        print(f"Error en {nombre}: {error}")
        return False
    else:
        return True

def crear_grafico(func_generacion, archivo, nombre, timeout=15):
    
    def crear_y_mostrar():
        fig = func_generacion()
        fig.show()
        fig.write_html(archivo)
        return fig
    
    if ejecutar_con_timeout(crear_y_mostrar, timeout, nombre):
        print(f"{nombre} generado correctamente")
        return True
    
    return False

print("Descargando datos de Apple y S&P 500...")
aapl = yf.Ticker("AAPL").history(start="2022-01-01", end="2023-12-31")
sp500 = yf.Ticker("^GSPC").history(start="2022-01-01", end="2023-12-31")

returns = aapl['Close'].pct_change().dropna()
sp500_returns = sp500['Close'].pct_change().dropna()

precio_actual = aapl['Close'].iloc[-1]
retorno_diario = returns.mean() * 100
retorno_total = (aapl['Close'].iloc[-1] / aapl['Close'].iloc[0] - 1) * 100
volatilidad = returns.std() * np.sqrt(252) * 100

sharpe = (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252))
cum_returns = (1 + returns).cumprod()
rolling_max = cum_returns.cummax()
drawdown = (cum_returns / rolling_max -1) * 100
max_drawdown = drawdown.min()
beta = returns.cov(sp500_returns) / sp500_returns.var()
yoy = (aapl['Close'].iloc[-1] - aapl['Close'].iloc[0]) / aapl['Close'].iloc[0] * 100

print("\n*** KPIs FINANCIEROS ***")
print(f"Precio actual: ${precio_actual:.2f}")
print(f"Retorno diario promedio: ${retorno_diario:.3f}%")
print(f"Retorno total: {retorno_total:.1f}%")
print(f"Volatilidad anualizada: {volatilidad:.1f}%")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Máximo drawdown: {max_drawdown:.1f}%")
print(f"Beta (vs S&P 500): {beta:.2f}%")
print(f"YoY (2022-2023): {yoy:.1f}%")

print("\n *** GENERANDO GRÁFICOS ***")
graficos_exitosos = 0

# 1.- Precio temporal
def grafica_precio():
    return px.line(aapl, x=aapl.index, y='Close', title='Precio Apple 2022-2023')

if crear_grafico(grafica_precio, "precio_apple.html", "Grafico precio"):
    graficos_exitosos += 1
time.sleep(2)

# 2.- Candlestick
def grafico_velas():
    fig = go.Figure(data=go.Candlestick(x=aapl.index, 
                                        open=aapl['Open'],
                                        high=aapl['High'],
                                        low=aapl['Low'],
                                        close=aapl['Close']))
    fig.update_layout(title='Gráfico de Velas - Apple')
    return fig

if crear_grafico(grafico_velas, "velas_apple.html", "Gráfico de Velas"):
    graficos_exitosos += 1

time.sleep(2)

# 3.- KPIs comparativo
def grafico_kpis():
    kpis_data = pd.DataFrame({
        'Métrica': ['Retorno Total', 'Volatilidad', 'Sharpe', 'Max Drawdown'],
        'Valor': [retorno_total, volatilidad, sharpe, max_drawdown]
    })
    return px.bar(kpis_data, x='Métrica', y='Valor', title='KPIs Financieros Apple')

if crear_grafico(grafico_kpis, "kpis_apple.html", "Gráficos de KPIs"):
    graficos_exitosos += 1

time.sleep(2)

# 4.- Histograma retorno
def grafico_histograma():
    returns_clean = (returns * 100)[(returns * 100 >= -10) & (returns * 100 <= 10)]
    fig = px.histogram(x=returns_clean, title='Distribución Retornos Diarios (%)', nbins=20)
    fig.update_layout(showlegend=False)
    return fig

if crear_grafico(grafico_histograma, "histograma_retornos.html", "Histograma de retornos"):
    graficos_exitosos += 1

time.sleep(2)

print(f"\n*** RESUMEN ***")
print(f"Gráficos generados: {graficos_exitosos}/4")
print("Archivos HTML creados: ")
print("- precio_apple.html")
print("- velas_apple.html")
print("- kpis_apple.html")
print("- histograma_retornos.html")
