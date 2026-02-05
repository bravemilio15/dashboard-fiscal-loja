"""
Dashboard Ejecutivo - Proyecto de Data Mining
Análisis Fiscal Loja 2020-2024
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append(str(Path(__file__).parent))
from utils.icons import icon, icon_text, MATERIAL_ICONS_CDN, Icons

# Configuración
st.set_page_config(
    page_title="Dashboard Fiscal Loja",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar Material Icons CDN
st.markdown(MATERIAL_ICONS_CDN, unsafe_allow_html=True)

# Función para cargar dataset
@st.cache_data
def load_default_dataset():
    try:
        dataset_path = Path(__file__).parent / "Dataset_Loja_Preprocesado.csv"
        df = pd.read_csv(dataset_path)
        
        # Crear FLAG_ES_CERO si no existe (0 = tributa, 1 = no tributa)
        if 'FLAG_ES_CERO' not in df.columns and 'VALOR_RECAUDADO' in df.columns:
            df['FLAG_ES_CERO'] = (df['VALOR_RECAUDADO'] == 0).astype(int)
            print(f"[DEBUG] FLAG_ES_CERO creado: {df['FLAG_ES_CERO'].value_counts().to_dict()}")
        
        # Renombrar columnas para compatibilidad
        if 'DESCRIPCION_ACT_ECONOMICA' in df.columns:
            df['ACTIVIDAD_ECONOMICA'] = df['DESCRIPCION_ACT_ECONOMICA']
            print(f"[DEBUG] ACTIVIDAD_ECONOMICA creada: {df['ACTIVIDAD_ECONOMICA'].nunique()} únicas")
        
        # Asegurar que tenemos la columna FECHA
        if 'FECHA' not in df.columns and 'ANIO' in df.columns and 'NOMBRE_MES' in df.columns:
            meses_map = {
                'Enero': '01', 'Febrero': '02', 'Marzo': '03', 'Abril': '04',
                'Mayo': '05', 'Junio': '06', 'Julio': '07', 'Agosto': '08',
                'Septiembre': '09', 'Octubre': '10', 'Noviembre': '11', 'Diciembre': '12'
            }
            df['FECHA'] = pd.to_datetime(
                df['ANIO'].astype(str) + '-' + df['NOMBRE_MES'].map(meses_map) + '-01'
            )
        
        return df, None
    except Exception as e:
        return None, str(e)

# Inicializar session_state
if 'df' not in st.session_state or st.session_state['df'] is None:
    df, error = load_default_dataset()
    if df is not None:
        st.session_state['df'] = df
        st.session_state['dataset_loaded'] = True
        st.session_state['dataset_name'] = "Dataset_Loja_Preprocesado.csv"
    else:
        st.session_state['df'] = None
        st.session_state['dataset_loaded'] = False

# Sidebar
st.sidebar.title("■ Dashboard Fiscal Loja")
st.sidebar.markdown("---")

st.sidebar.info("""
**Proyecto:** Análisis Fiscal Provincial

**Ubicación:** Loja, Ecuador

**Período:** 2020-2024

**Datos:** Recaudación SRI
""")

st.sidebar.markdown("---")

if st.session_state.get('dataset_loaded', False):
    df = st.session_state['df']
    total_rec = df['VALOR_RECAUDADO'].sum() if 'VALOR_RECAUDADO' in df.columns else 0
    st.sidebar.markdown(f"""
    <div style='text-align:center;padding:10px;background:#f0f2f6;border-radius:10px'>
        {icon('attach_money', 40, '#2ecc71')}<br>
        <h3 style='margin:5px 0'>${total_rec/1e6:.1f}M</h3>
        <small>Total Recaudado</small>
    </div>
    <br>
    <div style='text-align:center;padding:10px;background:#f0f2f6;border-radius:10px'>
        {icon('description', 40, '#3498db')}<br>
        <h3 style='margin:5px 0'>{len(df):,}</h3>
        <small>Total Registros</small>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PÁGINA PRINCIPAL: PANEL DE KPIs (IMPACTO DE NEGOCIO)
# ==============================================================================

st.markdown(f"# {icon_text(Icons.ANALYTICS, 'Panel de KPIs - Impacto de Negocio', 32, '#1f77b4')}", unsafe_allow_html=True)
st.markdown("### Indicadores Clave de Recaudación Fiscal")
st.markdown("---")

if not st.session_state.get('dataset_loaded', False):
    st.markdown(f":red[{icon(Icons.WARNING, 20, '#e74c3c')} No se pudo cargar el dataset]", unsafe_allow_html=True)
    st.stop()

df = st.session_state['df']

# SECCIÓN 1: TARJETAS DE INDICADORES CLAVE
st.markdown(f"## {icon_text(Icons.TARGET, 'Indicadores Clave de Rendimiento', 24, '#2ecc71')}", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_rec = df['VALOR_RECAUDADO'].sum()
    st.markdown(f"""
    <div style='text-align:center'>
        {icon('attach_money', 60, '#2ecc71')}<br>
        <h2 style='margin:0'>${total_rec/1e6:.1f}M</h2>
        <p>Monto Total Recaudado</p>
        <small>{len(df):,} registros</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if 'ANIO' in df.columns and 'VALOR_RECAUDADO' in df.columns:
        rec_anual = df.groupby('ANIO')['VALOR_RECAUDADO'].sum()
        if len(rec_anual) >= 2:
            crec_2020_2024 = ((rec_anual.iloc[-1] / rec_anual.iloc[0]) - 1) * 100
            st.markdown(f"""
            <div style='text-align:center'>
                {icon('trending_up', 60, '#3498db')}<br>
                <h2 style='margin:0'>{crec_2020_2024:.1f}%</h2>
                <p>Crecimiento Acumulado</p>
                <small>2020-2024</small>
            </div>
            """, unsafe_allow_html=True)

with col3:
    if 'FLAG_ES_CERO' in df.columns:
        total = len(df)
        tributan = (df['FLAG_ES_CERO'] == 0).sum()
        eficiencia = (tributan / total) * 100
        st.markdown(f"""
        <div style='text-align:center'>
            {icon('bolt', 60, '#f39c12')}<br>
            <h2 style='margin:0'>{eficiencia:.1f}%</h2>
            <p>Eficiencia de Recaudación</p>
            <small>{tributan:,} tributan</small>
        </div>
        """, unsafe_allow_html=True)

with col4:
    if 'CANTON' in df.columns:
        num_cantones = df['CANTON'].nunique()
        st.markdown(f"""
        <div style='text-align:center'>
            {icon('map', 60, '#9b59b6')}<br>
            <h2 style='margin:0'>{num_cantones}</h2>
            <p>Cobertura Territorial</p>
            <small>cantones de Loja</small>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# SECCIÓN 2: TENDENCIA HISTÓRICA
st.markdown(f"## {icon_text(Icons.TRENDING_UP, 'Evolución Histórica de Recaudación', 24, '#3498db')}", unsafe_allow_html=True)

if 'ANIO' in df.columns and 'VALOR_RECAUDADO' in df.columns:
    rec_anual = df.groupby('ANIO')['VALOR_RECAUDADO'].sum().reset_index()
    rec_anual['MILLONES'] = rec_anual['VALOR_RECAUDADO'] / 1e6
    
    fig = go.Figure()
    
    # Área con gradiente (SIN texto en el trace)
    fig.add_trace(go.Scatter(
        x=rec_anual['ANIO'],
        y=rec_anual['MILLONES'],
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#2ecc71', width=4),
        marker=dict(size=15, symbol='diamond', line=dict(width=2, color='white')),
        name='Recaudación Anual',
        hovertemplate='<b>%{x}</b><br>$%{y:.1f}M<extra></extra>'
    ))
    
    # Calcular el máximo para el eje Y con más margen
    max_val = rec_anual['MILLONES'].max()
    y_max = max_val * 1.25  # 25% más de espacio arriba
    
    fig.update_layout(
        title="<b>Tendencia de Recaudación 2020-2024: Salud Fiscal del Quinquenio</b>",
        xaxis_title="<b>Año</b>",
        yaxis_title="<b>Millones de Dólares ($)</b>",
        height=550,
        hovermode='x unified',
        template='plotly_white',
        showlegend=False,
        xaxis=dict(
            tickmode='linear',
            tick0=2020,
            dtick=1,
            tickformat='d'
        ),
        yaxis=dict(
            range=[0, y_max]
        ),
        margin=dict(t=100, b=60)
    )
    
    # Agregar anotaciones FUERA del gráfico
    for i, row in rec_anual.iterrows():
        fig.add_annotation(
            x=row['ANIO'],
            y=row['MILLONES'],
            text=f"<b>${row['MILLONES']:.1f}M</b>",
            showarrow=False,
            yshift=25,  # Desplazar 25px arriba del punto
            font=dict(size=13, color='#1e8449', family='Arial Black'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#2ecc71',
            borderwidth=1,
            borderpad=4
        )
    
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# SECCIÓN 3: ANÁLISIS GEOGRÁFICO (CAPITAL VS PERIFERIA)
st.markdown(f"## {icon_text(Icons.MAP, 'Análisis Geográfico: Capital vs. Periferia', 24, '#9b59b6')}", unsafe_allow_html=True)
st.markdown("")

# Fila superior: Gráfico de barras + Gráfico de dona
col1, col2 = st.columns([3, 2])

with col1:
    if 'CANTON' in df.columns and 'VALOR_RECAUDADO' in df.columns:
        # Ranking de cantones
        rec_canton = df.groupby('CANTON')['VALOR_RECAUDADO'].sum().sort_values(ascending=False).head(10)
        rec_canton_pct = (rec_canton / df['VALOR_RECAUDADO'].sum() * 100).round(1)
        
        fig = go.Figure()
        
        # Colorear diferente la capital
        colors = ['#e74c3c' if canton == 'LOJA' else '#3498db' for canton in rec_canton.index]
        
        fig.add_trace(go.Bar(
            y=rec_canton.index,
            x=rec_canton.values / 1e6,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=3)
            ),
            text=[f"<b>{pct}%</b>" for pct in rec_canton_pct],
            textposition='outside',
            textfont=dict(size=15, family='Arial Black', color='#000000'),
            hovertemplate='<b>%{y}</b><br>Recaudación: $%{x:.1f}M<br>Porcentaje: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>Ranking de Cantones: Concentración Fiscal</b>",
            xaxis_title="<b>Millones de Dólares ($)</b>",
            yaxis_title="",
            height=550,
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=18, color='#2C3E50', family='Arial Black'),
            xaxis=dict(gridcolor='#E8E8E8', range=[0, rec_canton.max() / 1e6 * 1.15]),
            margin=dict(l=120, r=100, t=60, b=60)
        )
        
        st.plotly_chart(fig, width='stretch')

with col2:
    # Gráfico de dona
    if 'CANTON' in df.columns:
        rec_loja = df[df['CANTON'] == 'LOJA']['VALOR_RECAUDADO'].sum()
        rec_otros = df[df['CANTON'] != 'LOJA']['VALOR_RECAUDADO'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['CAPITAL<br>(Loja)', 'PERIFERIA<br>(Otros)'],
            values=[rec_loja, rec_otros],
            hole=0.6,
            marker=dict(colors=['#E74C3C', '#3498DB'], line=dict(color='white', width=4)),
            textinfo='percent',
            textfont=dict(size=20, color='#000000', family='Arial Black'),
            textposition='inside',
            hovertemplate='<b>%{label}</b><br>%{percent}<br>$%{value:,.0f}<extra></extra>'
        )])
        
        fig.update_layout(
            title="<b>Distribución Geográfica</b>",
            height=550,
            template='plotly_white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=16, color='#2C3E50', family='Arial Black'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.05,
                xanchor="center",
                x=0.5,
                font=dict(size=13, color='#000000', family='Arial Black')
            ),
            margin=dict(l=10, r=10, t=60, b=50)
        )
        
        st.plotly_chart(fig, width='stretch')

# Fila inferior: Métricas de concentración en 4 columnas
st.markdown("#### 📊 Indicadores de Concentración")
if 'CANTON' in df.columns:
    rec_loja = df[df['CANTON'] == 'LOJA']['VALOR_RECAUDADO'].sum()
    rec_total = df['VALOR_RECAUDADO'].sum()
    pct_capital = (rec_loja / rec_total) * 100
    
    rec_periferia = rec_total - rec_loja
    pct_periferia = 100 - pct_capital
    ratio = rec_loja / rec_periferia if rec_periferia > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Capital (Loja)", f"{pct_capital:.1f}%", delta=f"${rec_loja/1e6:.1f}M")
    
    with col2:
        st.metric("Periferia", f"{pct_periferia:.1f}%", delta=f"${rec_periferia/1e6:.1f}M")
    
    with col3:
        st.metric("Ratio Capital/Periferia", f"{ratio:.1f}x")
    
    with col4:
        if pct_capital > 80:
            st.success("✅ Alta concentración")
        else:
            st.info("ℹ️ Distribución equilibrada")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Dashboard Ejecutivo - Analisis Fiscal Loja 2020-2024</strong></p>
    <p>Proyecto de Data Mining | Universidad Nacional de Loja</p>
</div>
""", unsafe_allow_html=True)

