import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize   # Optimización

# ------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Enfoque de Mínima Varianza y su Aplicación en Portafolios Financieros durante la Pandemia",
    layout="wide"
)

# Paleta USTA
USTA_PALETTE = {
    "primary": "#0a2f6b",
    "secondary": "#1d4f91",
    "accent": "#f9a602",
    "teal": "#1c9c9c",
    "neutral_light": "#f5f7fa",
    "neutral_mid": "#60738a",
    "neutral_dark": "#17213c",
}

PLOTLY_COLOR_SEQUENCE = [
    USTA_PALETTE["primary"],
    USTA_PALETTE["secondary"],
    USTA_PALETTE["teal"],
    USTA_PALETTE["accent"],
]


def inject_brand_theme() -> None:
    """Aplica estilos visuales tipo USTA."""
    st.markdown(
        f"""
        <style>
            :root {{
                --usta-primary: {USTA_PALETTE['primary']};
                --usta-secondary: {USTA_PALETTE['secondary']};
                --usta-accent: {USTA_PALETTE['accent']};
                --usta-teal: {USTA_PALETTE['teal']};
                --usta-neutral-light: {USTA_PALETTE['neutral_light']};
                --usta-neutral-mid: {USTA_PALETTE['neutral_mid']};
                --usta-neutral-dark: {USTA_PALETTE['neutral_dark']};
            }}

            .stApp {{
                background: linear-gradient(180deg, rgba(10,47,107,0.06) 0%, rgba(28,156,156,0.03) 100%);
                color: var(--usta-neutral-dark);
            }}

            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, rgba(10,47,107,0.96) 0%, rgba(29,79,145,0.93) 40%, rgba(28,156,156,0.9) 100%);
                color: #ffffff;
                border-right: 0;
            }}

            section[data-testid="stSidebar"] * {{
                color: #ffffff !important;
            }}

            section[data-testid="stSidebar"] label {{
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}

            section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
            section[data-testid="stSidebar"] .stNumberInput input,
            section[data-testid="stSidebar"] .stTextInput input {{
                background: rgba(255,255,255,0.9);
                border: 1px solid rgba(255,255,255,0.45);
                border-radius: 10px;
                color: var(--usta-neutral-dark) !important;
            }}

            .block-container {{
                padding-top: 1.8rem;
            }}

            h1, h2, h3, h4 {{
                color: var(--usta-primary) !important;
                font-family: "Montserrat", "Segoe UI", sans-serif;
            }}

            .stMetric {{
                background: rgba(255,255,255,0.86);
                padding: 1rem;
                border-radius: 14px;
                border: 1px solid rgba(10,47,107,0.12);
                box-shadow: 0 8px 24px rgba(23,33,60,0.07);
            }}

            div[data-testid="stMetricValue"] {{
                color: var(--usta-primary) !important;
                font-weight: 700;
            }}

            div[data-testid="stMetricLabel"] {{
                color: var(--usta-neutral-mid) !important;
            }}

            .stDataFrame {{
                border-radius: 16px;
                box-shadow: 0 10px 26px rgba(15,33,58,0.08);
            }}

            .stDownloadButton button {{
                background: linear-gradient(135deg, var(--usta-accent), #ffd166);
                color: var(--usta-neutral-dark) !important;
                font-weight: 700;
                border-radius: 999px;
                border: none;
                padding: 0.55rem 1.6rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig):
    """Aplica tema de colores USTA a una figura Plotly."""
    fig.update_layout(
        font=dict(
            family="Montserrat, 'Segoe UI', sans-serif",
            color=USTA_PALETTE["neutral_dark"],
        ),
        paper_bgcolor=USTA_PALETTE["neutral_light"],
        plot_bgcolor="#ffffff",
        title=dict(font=dict(color=USTA_PALETTE["primary"], size=18)),
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(10,47,107,0.18)",
            borderwidth=0.5,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(10,47,107,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(10,47,107,0.08)")
    return fig


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convierte un DataFrame en CSV UTF-8 con BOM (ideal para Excel)."""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


inject_brand_theme()

# ------------------------------------------------------------
# CONFIG GLOBAL
# ------------------------------------------------------------
TICKERS = ["EC", "CIB", "IESFY", "IEF"]    # 4 activos del portafolio
BENCHMARK = "IEF"                          # Benchmark para Beta/CAPM
RF = 0.03                                  # Tasa libre de riesgo anual (3%)

SUBMUESTRAS = {
    "2015-2019": ("2015-01-01", "2019-12-31"),
    "2016-2020": ("2016-01-01", "2020-12-31"),
    "2020":      ("2020-01-01", "2020-12-31"),
    "2020-2023": ("2020-01-01", "2023-12-31"),
    "2023":      ("2023-01-01", "2023-12-31"),
}

# ------------------------------------------------------------
# FUNCIONES
# ------------------------------------------------------------

def descargar_precios_close(tickers, inicio, fin):
    """Descarga precios de cierre y devuelve DataFrame con columnas = tickers."""
    df = yf.download(tickers, start=inicio, end=fin, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError("No se descargaron datos de Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in lvl0:
            close = df.xs("Close", axis=1, level=0)
        elif "Adj Close" in lvl0:
            close = df.xs("Adj Close", axis=1, level=0)
        else:
            raise ValueError("No se encontró 'Close' ni 'Adj Close' en las columnas.")
    else:
        if "Close" in df.columns:
            close = df[["Close"]]
        elif "Adj Close" in df.columns:
            close = df[["Adj Close"]]
        else:
            raise ValueError("No se encontró 'Close' ni 'Adj Close' en las columnas.")
        if len(tickers) == 1:
            close.columns = [tickers[0]]

    cols_presentes = [t for t in tickers if t in close.columns]
    close = close[cols_presentes]

    return close


def calcular_metricas(returns, benchmark, rf):
    """Retorno anual, volatilidad, beta, Sharpe, Treynor, CAPM para cada activo."""
    resultados = []
    bench = returns[benchmark]
    mu_bench = bench.mean() * 252
    var_bench = bench.var()

    for col in returns.columns:
        r = returns[col]
        mu = r.mean() * 252
        sigma = r.std() * np.sqrt(252)

        cov = np.cov(r, bench)[0, 1]
        beta = cov / var_bench if var_bench > 0 else np.nan

        sharpe = (mu - rf) / sigma if sigma > 0 else np.nan
        treynor = (mu - rf) / beta if (beta not in [0, np.nan]) else np.nan
        capm = rf + beta * (mu_bench - rf) if not np.isnan(beta) else np.nan

        resultados.append(
            {
                "Ticker": col,
                "Retorno anual (%)": mu * 100,
                "Volatilidad anual (%)": sigma * 100,
                "Beta vs IEF": beta,
                "Sharpe": sharpe,
                "Treynor": treynor,
                "CAPM esperado (%)": capm * 100,
            }
        )

    return pd.DataFrame(resultados)


def portafolio_min_var(returns):
    """
    Portafolio de mínima varianza con:
      - sum(w) = 1
      - 0 <= w_i <= 1  (sin posiciones cortas)
    """
    cov = returns.cov() * 252
    n = cov.shape[0]

    def var_port(w):
        return float(w @ cov.values @ w)

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(var_port, w0, method="SLSQP", bounds=bounds, constraints=cons)

    if not res.success:
        raise RuntimeError("Falló la optimización de mínima varianza: " + res.message)

    w_opt = res.x
    var_min = var_port(w_opt)

    pesos = pd.Series(w_opt, index=returns.columns)
    vol = np.sqrt(var_min)

    return pesos, vol


def frontera_eficiente(returns, n_points=500):
    """Portafolios aleatorios para aproximar la frontera eficiente."""
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(mu)

    resultados = []
    for _ in range(n_points):
        w = np.random.dirichlet(np.ones(n))
        r = w @ mu.values
        v = np.sqrt(w @ cov.values @ w)
        resultados.append((v, r))
    return pd.DataFrame(resultados, columns=["Volatilidad", "Retorno"])


def mostrar_contexto():
    """Muestra periodo y activo en forma de métricas bonitas."""
    col_a, col_b = st.columns(2)
    col_a.metric("Periodo seleccionado", f"{inicio} → {fin}")
    col_b.metric("Activo seleccionado", activo_sel)


# ------------------------------------------------------------
# BARRA LATERAL DE FILTROS
# ------------------------------------------------------------
with st.sidebar:
    st.title("Información del Proyecto")
    st.markdown(
        """
Nombre del Proyecto  
Enfoque de Mínima Varianza y su Aplicación en Portafolios Financieros durante la Pandemia  

Integrantes del equipo  
- Yeimy Alarcón  
- Camilo Castillo  
- Lina Rozo
        """
    )
    st.markdown("---")

    st.title("Filtros")

    activo_opciones = ["Todos"] + TICKERS
    activo_sel = st.selectbox("Activo:", options=activo_opciones, index=0)

    periodo = st.selectbox("Periodo:", list(SUBMUESTRAS.keys()), index=0)

inicio, fin = SUBMUESTRAS[periodo]

st.title(
    "Dashboard Enfoque de Mínima Varianza y su Aplicación en Portafolios Financieros durante la Pandemia"
)

# ------------------------------------------------------------
# DESCARGA DE DATOS
# ------------------------------------------------------------
with st.spinner("Descargando datos de Yahoo Finance..."):
    precios = descargar_precios_close(TICKERS, inicio, fin)

# Subset por activo si el filtro no es "Todos"
if activo_sel != "Todos":
    precios_filtrado = precios[[activo_sel]].copy()
else:
    precios_filtrado = precios.copy()

# Rendimientos (para toda la canasta, no solo el filtro)
retornos = precios.pct_change().dropna()

# ------------------------------------------------------------
# PESTAÑAS (Datos de últimas)
# ------------------------------------------------------------
tab1, tab2, tab3, tab_datos = st.tabs(
    [
        "Serie de precios por activo",
        "Precio + SMA20/EMA20",
        "Análisis",
        "Datos (precios de cierre)",
    ]
)

# ------------------------------------------------------------
# TAB 1: SERIES DE PRECIOS
# ------------------------------------------------------------
with tab1:
    mostrar_contexto()
    st.subheader("📈 Serie de precios por activo")

    df_plot = precios_filtrado.copy()
    df_plot.index.name = "Fecha"
    df_plot_reset = df_plot.reset_index()

    if activo_sel == "Todos":
        df_long = df_plot_reset.melt(
            id_vars="Fecha", var_name="Activo", value_name="Precio"
        )
        fig_prices = px.line(
            df_long,
            x="Fecha",
            y="Precio",
            color="Activo",
            title="Serie de precios por activo",
            color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
        )
        fig_prices = apply_plotly_theme(fig_prices)
        st.plotly_chart(fig_prices, use_container_width=True)
    else:
        col = activo_sel
        fig_single = px.line(
            df_plot_reset,
            x="Fecha",
            y=col,
            title=f"Serie de precios – {col}",
            color_discrete_sequence=[USTA_PALETTE["primary"]],
        )
        fig_single = apply_plotly_theme(fig_single)
        st.plotly_chart(fig_single, use_container_width=True)

# ------------------------------------------------------------
# TAB 2: SMA Y EMA
# ------------------------------------------------------------
with tab2:
    mostrar_contexto()
    st.subheader("📉 Precio, SMA 20 y EMA 20")

    window = 20

    if activo_sel == "Todos":
        activos_iter = precios.columns
    else:
        activos_iter = [activo_sel]

    for col in activos_iter:
        df_aux = precios[[col]].copy()
        df_aux["SMA20"] = df_aux[col].rolling(window=window).mean()
        df_aux["EMA20"] = df_aux[col].ewm(span=window, adjust=False).mean()
        df_aux.index.name = "Fecha"
        df_aux = df_aux.reset_index()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df_aux["Fecha"], y=df_aux[col], name=f"Precio {col}", mode="lines")
        )
        fig.add_trace(
            go.Scatter(x=df_aux["Fecha"], y=df_aux["SMA20"], name="SMA 20", mode="lines")
        )
        fig.add_trace(
            go.Scatter(x=df_aux["Fecha"], y=df_aux["EMA20"], name="EMA 20", mode="lines")
        )

        fig.update_layout(title=f"{col} – Precio, SMA20 y EMA20")
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TAB 3: ANÁLISIS
# ------------------------------------------------------------
with tab3:
    mostrar_contexto()
    # ---- Métricas ----
    st.subheader("📌 Métricas por Activo")

    tabla_metricas = calcular_metricas(retornos, BENCHMARK, RF)

    if activo_sel != "Todos":
        tabla_mostrar = tabla_metricas[tabla_metricas["Ticker"] == activo_sel]
        if tabla_mostrar.empty:
            tabla_mostrar = tabla_metricas
    else:
        tabla_mostrar = tabla_metricas

    st.dataframe(tabla_mostrar, use_container_width=True)

    st.markdown("---")

    # ---- Matriz de Covarianzas (ANUALIZADA) ----
    st.subheader("📊 Matriz de Covarianzas (anualizada)")
    cov_annual = retornos.cov() * 252
    cov_annual_round = cov_annual.round(6)
    st.dataframe(cov_annual_round, use_container_width=True)
    st.caption("Matriz de covarianzas de los retornos diarios, anualizada multiplicando por 252.")

    st.markdown("---")

    # ---- Portafolio mínima varianza ----
    st.subheader("📉 Portafolio de Mínima Varianza")

    pesos_min, vol_min = portafolio_min_var(retornos)
    ret_port_min = float((retornos.mean() * 252 @ pesos_min))

    st.write("### Pesos óptimos del portafolio:")
    tabla_pesos = pd.DataFrame(
        {"Activo": pesos_min.index, "Peso": pesos_min.values}
    )
    st.dataframe(tabla_pesos, use_container_width=True)

    # KPIs DESPUÉS de la tabla de pesos
    col_kpi1, col_kpi2 = st.columns(2)
    col_kpi1.metric(
        label="Retorno esperado del portafolio (anual, mínima varianza)",
        value=f"{ret_port_min:.4f}"
    )
    col_kpi2.metric(
        label="Volatilidad del portafolio (anual, mínima varianza)",
        value=f"{vol_min:.4f}"
    )

    st.markdown("---")

    # ---- Frontera eficiente ----
    st.subheader("📈 Frontera Eficiente")

    front = frontera_eficiente(retornos)

    fig_front = px.scatter(
        front,
        x="Volatilidad",
        y="Retorno",
        title=f"Frontera eficiente – {periodo}",
        opacity=0.6,
    )
    fig_front.add_scatter(
        x=[vol_min],
        y=[ret_port_min],
        mode="markers",
        marker=dict(size=11),
        name="Portafolio mínima varianza",
    )

    fig_front = apply_plotly_theme(fig_front)
    st.plotly_chart(fig_front, use_container_width=True)

# ------------------------------------------------------------
# TAB DATOS: precios de cierre + descarga (ÚLTIMA)
# ------------------------------------------------------------
with tab_datos:
    mostrar_contexto()
    st.subheader("📂 Precios de cierre usados")

    df_export = precios_filtrado.copy()
    df_export.index.name = "Fecha"
    df_export_reset = df_export.reset_index()

    st.dataframe(df_export_reset, use_container_width=True)

    csv_bytes = convert_df_to_csv(df_export_reset)
    nombre_activo = "Todos" if activo_sel == "Todos" else activo_sel
    file_name = f"precios_cierre_{nombre_activo}_{periodo}.csv"

    st.download_button(
        label="⬇ Descargar datos",
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
    )