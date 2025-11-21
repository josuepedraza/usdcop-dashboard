import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="USD/COP – Modelo SARIMAX + GARCH",
    layout="wide"
)

# Paleta USTA (igual que tu dashboard original)
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
    "#00796B",   # cl_f
    "#4A148C",   # dxy
    "#f9a701",   # gc_f
    "#C62828",   # vix
    "#1565C0",   # tnx
]

DATA_DIR = Path("datos_modelo")  # carpeta donde guardaste los CSV


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
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(10,47,107,0.18)",
            borderwidth=0.5,
            font=dict(color="#0A2F6B", size=13),   # 👈 color de texto visible
        ),

    )
    # Ejes con labels visibles
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(10,47,107,0.08)",
        title_font=dict(color=USTA_PALETTE["neutral_dark"]),
        tickfont=dict(color=USTA_PALETTE["neutral_dark"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(10,47,107,0.08)",
        title_font=dict(color=USTA_PALETTE["neutral_dark"]),
        tickfont=dict(color=USTA_PALETTE["neutral_dark"]),
    )
    return fig


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convierte un DataFrame en CSV UTF-8 con BOM (ideal para Excel)."""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


@st.cache_data
def cargar_retornos():
    df = pd.read_csv(DATA_DIR / "retornos_usdcop.csv", parse_dates=["Fecha"])
    df = df.set_index("Fecha")
    return df


@st.cache_data
def cargar_pronostico():
    df = pd.read_csv(DATA_DIR / "pronostico_usdcop.csv", parse_dates=["Fecha"])
    return df


@st.cache_data
def cargar_residuos():
    df = pd.read_csv(DATA_DIR / "residuos_usdcop.csv", parse_dates=["Fecha"])
    return df


def calcular_metricas_pronostico(df_pron):
    """Calcula RMSE, MAE y MAPE como en tu código de R."""
    errores = (df_pron["real"] - df_pron["pronostico"]) * 100
    rmse = float(np.sqrt(np.mean(errores**2)))
    mae = float(np.mean(np.abs(errores)))
    mape = float(np.mean(np.abs(errores / df_pron["real"])) * 100)
    return rmse, mae, mape


# -------------------------------------------------------------------
# APLICAR TEMA
# -------------------------------------------------------------------
inject_brand_theme()

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.title("Información del Proyecto")
    st.markdown(
        """
**Modelo SARIMAX + GARCH para el tipo de cambio USD/COP**  

Integrantes del equipo  
- Carlos Diaz
- Thomas Orjuela 
- Josue Pedraza

Fuente de datos: Yahoo Finance (procesado en R).  

Modelado en R (SARIMAX + GARCH) y visualizado en este dashboard con Streamlit.
        """
    )
    st.markdown("---")

    st.title("Filtros")

    # Cargamos retornos para tomar fechas mínima y máxima
    df_ret = cargar_retornos()
    fecha_min = df_ret.index.min()
    fecha_max = df_ret.index.max()

    rango_fechas = st.date_input(
        "Rango de fechas para las gráficas:",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )

    # Selector de serie para graficar
    columnas = ["usd_cop", "cl_f", "dxy", "gc_f", "vix", "tnx"]
    serie_sel = st.selectbox(
        "Serie principal a mostrar:",
        options=columnas,
        index=0
    )

# Filtrar según rango de fechas
inicio, fin = rango_fechas
df_retornos = df_ret.loc[(df_ret.index >= pd.to_datetime(inicio)) &
                         (df_ret.index <= pd.to_datetime(fin))]

df_pron = cargar_pronostico()
df_resid = cargar_residuos()

st.title("Dashboard – USD/COP: Retornos, Pronósticos y Diagnóstico")

# -------------------------------------------------------------------
# TABS PRINCIPALES
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Retornos",
        "Pronóstico vs Real",
        "Diagnóstico de residuos",
        "Datos"
    ]
)

# -------------------------------------------------------------------
# TAB 1: RETORNOS
# -------------------------------------------------------------------
with tab1:
    st.subheader("📈 Retornos diarios")

    # 1. Serie principal (ej. usd_cop)
    df_plot = df_retornos.reset_index()
    fig_main = px.line(
        df_plot,
        x="Fecha",
        y=serie_sel,
        title=f"Retornos diarios – {serie_sel}",
        color_discrete_sequence=[USTA_PALETTE["primary"]],
    )
    fig_main = apply_plotly_theme(fig_main)
    st.plotly_chart(fig_main, use_container_width=True)

    # 2. Exógenas (todas menos usd_cop)
    st.markdown("### Retornos de variables exógenas")
    exogenas_cols = [c for c in df_retornos.columns if c != "usd_cop"]
    df_long = df_plot.melt(
        id_vars="Fecha",
        value_vars=exogenas_cols,
        var_name="Serie",
        value_name="Retorno"
    )
    fig_exog = px.line(
        df_long,
        x="Fecha",
        y="Retorno",
        color="Serie",
        title="Retornos de variables exógenas",
        color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
    )
    fig_exog = apply_plotly_theme(fig_exog)
    st.plotly_chart(fig_exog, use_container_width=True)

# -------------------------------------------------------------------
# TAB 2: PRONÓSTICO VS REAL
# -------------------------------------------------------------------
with tab2:
    st.subheader("📊 Comparación: Real vs Pronóstico (USD/COP – retornos)")

    fig_fc = px.line(
        df_pron,
        x="Fecha",
        y=["real", "pronostico"],
        labels={"value": "Retorno", "variable": "Serie"},
        title="Serie real vs pronósticos (modelo GARCH sobre SARIMAX)",
        color_discrete_sequence=[USTA_PALETTE["primary"], USTA_PALETTE["accent"]],
    )
    fig_fc = apply_plotly_theme(fig_fc)
    st.plotly_chart(fig_fc, use_container_width=True)

    # Métricas de error
    rmse, mae, mape = calcular_metricas_pronostico(df_pron)

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE (x100)", f"{rmse:.4f}")
    col2.metric("MAE (x100)", f"{mae:.4f}")
    col3.metric("MAPE (%)", f"{mape:.2f}")

# -------------------------------------------------------------------
# TAB 3: DIAGNÓSTICO DE RESIDUOS
# -------------------------------------------------------------------
with tab3:
    st.subheader("🔍 Residuos estandarizados del modelo")

    # Serie de residuos
    fig_res = px.line(
        df_resid,
        x="Fecha",
        y="Residual",
        title="Residuos estandarizados en el tiempo",
        color_discrete_sequence=[USTA_PALETTE["primary"]],
    )
    fig_res = apply_plotly_theme(fig_res)
    st.plotly_chart(fig_res, use_container_width=True)

    col_a, col_b = st.columns(2)

    # Histograma
    with col_a:
        fig_hist = px.histogram(
            df_resid,
            x="Residual",
            nbins=40,
            title="Histograma de residuos estandarizados",
            color_discrete_sequence=[USTA_PALETTE["secondary"]],
        )
        fig_hist = apply_plotly_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    # QQ-plot aproximado (contra normal estándar)
    with col_b:
        resid = df_resid["Residual"].dropna().values
        resid_sorted = np.sort(resid)
        n = len(resid_sorted)

        # cuantiles teóricos de una normal estándar
        probs = (np.arange(1, n + 1) - 0.5) / n
        teorico = np.quantile(np.random.normal(size=10_000), probs)

        df_qq = pd.DataFrame(
            {"Teorico": teorico, "Muestral": resid_sorted}
        )

        # Scatter SIN trendline
        fig_qq = px.scatter(
            df_qq,
            x="Teorico",
            y="Muestral",
            title="QQ-Plot aproximado (residuos vs Normal)",
            color_discrete_sequence=[USTA_PALETTE["accent"]],
        )

        # Línea de 45°
        min_val = min(df_qq["Teorico"].min(), df_qq["Muestral"].min())
        max_val = max(df_qq["Teorico"].max(), df_qq["Muestral"].max())

        fig_qq.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Línea 45°",
                line=dict(dash="dash"),
            )
        )

        fig_qq = apply_plotly_theme(fig_qq)
        st.plotly_chart(fig_qq, use_container_width=True)

# -------------------------------------------------------------------
# TAB 4: DATOS
# -------------------------------------------------------------------
with tab4:
    st.subheader("📂 Datos usados en el dashboard")

    st.markdown("### Retornos diarios (todas las series)")
    df_export_ret = df_retornos.reset_index()
    st.dataframe(df_export_ret, use_container_width=True)

    csv_ret = convert_df_to_csv(df_export_ret)
    st.download_button(
        label="⬇ Descargar retornos filtrados",
        data=csv_ret,
        file_name="retornos_filtrados_usdcop.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("### Real vs Pronóstico")
    st.dataframe(df_pron, use_container_width=True)

    csv_pron = convert_df_to_csv(df_pron)
    st.download_button(
        label="⬇ Descargar tabla de pronósticos",
        data=csv_pron,
        file_name="pronostico_usdcop.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("### Residuos estandarizados")
    st.dataframe(df_resid, use_container_width=True)

    csv_res = convert_df_to_csv(df_resid)
    st.download_button(
        label="⬇ Descargar residuos",
        data=csv_res,
        file_name="residuos_usdcop.csv",
        mime="text/csv",
    )
