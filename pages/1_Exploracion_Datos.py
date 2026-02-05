"""
Página 1: Análisis Exploratorio
Visualizaciones automáticas del dataset preprocesado
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.icons import icon, icon_text, MATERIAL_ICONS_CDN, Icons

st.set_page_config(page_title="Análisis Exploratorio", page_icon="📊", layout="wide")

# Cargar Material Icons CDN
st.markdown(MATERIAL_ICONS_CDN, unsafe_allow_html=True)

st.markdown(f"# {icon_text(Icons.ANALYTICS, 'Análisis Exploratorio de Datos', 32, '#1f77b4')}", unsafe_allow_html=True)
st.markdown("### Provincia de Loja - Recaudación Fiscal 2020-2024")
st.markdown("---")

# Verificar que el dataset esté cargado
if not st.session_state.get('dataset_loaded', False) or st.session_state['df'] is None:
    st.markdown(f":red[{icon(Icons.WARNING, 20, '#e74c3c')} Dataset no disponible. Recarga la página principal.]", unsafe_allow_html=True)
    st.stop()

df = st.session_state['df']

# Crear columnas necesarias si no existen
if 'FLAG_ES_CERO' not in df.columns and 'VALOR_RECAUDADO' in df.columns:
    df['FLAG_ES_CERO'] = (df['VALOR_RECAUDADO'] == 0).astype(int)
    st.session_state['df'] = df
    print(f"[DEBUG Exploracion] FLAG_ES_CERO creado: {df['FLAG_ES_CERO'].value_counts().to_dict()}")

if 'ACTIVIDAD_ECONOMICA' not in df.columns and 'DESCRIPCION_ACT_ECONOMICA' in df.columns:
    df['ACTIVIDAD_ECONOMICA'] = df['DESCRIPCION_ACT_ECONOMICA']
    st.session_state['df'] = df
    print(f"[DEBUG Exploracion] ACTIVIDAD_ECONOMICA creada con {df['ACTIVIDAD_ECONOMICA'].nunique()} únicas")

# Sección 1: Estadísticas Descriptivas
st.header("1. Estadísticas Descriptivas Generales")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total de Registros", f"{len(df):,}")

with col2:
    if 'VALOR_RECAUDADO' in df.columns:
        promedio = df['VALOR_RECAUDADO'].mean()
        st.metric("Recaudación Promedio", f"${promedio:,.0f}")

with col3:
    if 'FLAG_ES_CERO' in df.columns:
        tributan = (df['FLAG_ES_CERO'] == 0).sum()
        st.metric("Contribuyentes que Tributan", f"{tributan:,}")

with col4:
    if 'CANTON' in df.columns:
        st.metric("Cantones", df['CANTON'].nunique())

with col5:
    if 'ACTIVIDAD_ECONOMICA' in df.columns:
        st.metric("Actividades Económicas", df['ACTIVIDAD_ECONOMICA'].nunique())

st.markdown("---")

# Sección 2: Distribución de la Variable Objetivo
st.header("2. Distribución de Contribuyentes")

if 'FLAG_ES_CERO' in df.columns:
    print(f"[DEBUG] FLAG_ES_CERO encontrado. Valores: {df['FLAG_ES_CERO'].value_counts().to_dict()}")
    # Estadísticas
    dist = df['FLAG_ES_CERO'].value_counts()
    total = len(df)
    tributan = dist.get(0, 0)
    no_tributan = dist.get(1, 0)
    ratio = tributan / no_tributan if no_tributan > 0 else 0
    
    # Métricas en fila
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Registros", f"{total:,}")
    
    with col2:
        st.metric("✅ Tributan", f"{tributan:,}", delta=f"{tributan/total*100:.1f}%")
    
    with col3:
        st.metric("❌ No Tributan", f"{no_tributan:,}", delta=f"{no_tributan/total*100:.1f}%")
    
    with col4:
        st.metric("Ratio", f"{ratio:.2f}:1")
    
    # Visualización mejorada
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de dona simple
        fig = px.pie(
            values=[tributan, no_tributan],
            names=['Tributan (0)', 'No Tributan (1)'],
            title="Distribución: Contribuyentes que Tributan vs No Tributan",
            hole=0.4,
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gráfico de barras
        fig = px.bar(
            x=['Tributan (0)', 'No Tributan (1)'],
            y=[tributan, no_tributan],
            title="Comparación de Contribuyentes",
            labels={'x': 'Categoría', 'y': 'Cantidad'},
            color=['Tributan', 'No Tributan'],
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            text=[tributan, no_tributan]
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    if ratio > 3:
        st.warning("⚠️ Dataset desbalanceado (clase mayoritaria > 3x)")
    else:
        st.success("✅ Dataset balanceado")
else:
    st.error("[ERROR] No se encontró la columna FLAG_ES_CERO")
    print(f"[DEBUG] Columnas disponibles: {df.columns.tolist()}")

st.markdown("---")

# Sección 3: Análisis Temporal
st.header("3. Evolución Temporal de la Recaudación")

if 'ANIO' in df.columns and 'VALOR_RECAUDADO' in df.columns:
    # Recaudación por año
    recaudacion_anual = df.groupby('ANIO')['VALOR_RECAUDADO'].agg(['sum', 'mean', 'count']).reset_index()
    recaudacion_anual['sum_millones'] = recaudacion_anual['sum'] / 1e6
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de línea
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recaudacion_anual['ANIO'],
            y=recaudacion_anual['sum_millones'],
            mode='lines+markers',
            name='Recaudación Total',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Recaudación Total por Año (Millones $)",
            xaxis_title="Año",
            yaxis_title="Recaudación (Millones $)",
            height=400,
            hovermode='x unified',
            xaxis=dict(tickformat='d')  # Sin separadores de miles
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gráfico de barras con cantidad de registros
        fig = px.bar(
            recaudacion_anual,
            x='ANIO',
            y='count',
            title="Cantidad de Registros por Año",
            labels={'ANIO': 'Año', 'count': 'Número de Registros'},
            color='count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            xaxis=dict(tickformat='d')  # Sin separadores de miles
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla resumen
    st.markdown("### Resumen Anual")
    resumen_display = recaudacion_anual[['ANIO', 'sum_millones', 'count']].copy()
    resumen_display.columns = ['Año', 'Recaudación Total (Millones $)', 'Número de Registros']
    resumen_display['Recaudación Total (Millones $)'] = resumen_display['Recaudación Total (Millones $)'].round(2)
    st.dataframe(resumen_display, use_container_width=True, hide_index=True)

st.markdown("---")

# Sección 4: Distribución por Cantón
st.header("4. Análisis Geográfico: Distribución por Cantón")

if 'CANTON' in df.columns and 'VALOR_RECAUDADO' in df.columns:
    recaudacion_canton = df.groupby('CANTON')['VALOR_RECAUDADO'].sum().reset_index()
    recaudacion_canton.columns = ['CANTON', 'TOTAL_RECAUDADO']
    recaudacion_canton = recaudacion_canton.sort_values('TOTAL_RECAUDADO', ascending=False)
    recaudacion_canton['PORCENTAJE'] = (recaudacion_canton['TOTAL_RECAUDADO'] / recaudacion_canton['TOTAL_RECAUDADO'].sum() * 100).round(2)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de barras horizontales
        fig = px.bar(
            recaudacion_canton.head(10),
            y='CANTON',
            x='TOTAL_RECAUDADO',
            orientation='h',
            title="Top 10 Cantones por Recaudación Total",
            labels={'TOTAL_RECAUDADO': 'Recaudación Total ($)', 'CANTON': 'Cantón'},
            text='PORCENTAJE',
            color='TOTAL_RECAUDADO',
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gráfico de pastel para top 5
        fig = px.pie(
            recaudacion_canton.head(5),
            values='TOTAL_RECAUDADO',
            names='CANTON',
            title="Top 5 Cantones (% Recaudación)",
            hole=0.4
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar imagen guardada del notebook
    fig_path = Path(__file__).parent.parent / "Fig_7_Canton.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Análisis de Cantones (del notebook)", use_container_width=True)

st.markdown("---")

# Sección 5: Actividad Económica
st.header("5. Análisis Sectorial: Actividades Económicas")

print(f"[DEBUG] Verificando ACTIVIDAD_ECONOMICA: {'ACTIVIDAD_ECONOMICA' in df.columns}")
print(f"[DEBUG] Verificando DESCRIPCION_ACT_ECONOMICA: {'DESCRIPCION_ACT_ECONOMICA' in df.columns}")

if 'ACTIVIDAD_ECONOMICA' in df.columns and 'VALOR_RECAUDADO' in df.columns:
    print(f"[DEBUG] Procesando actividades económicas: {df['ACTIVIDAD_ECONOMICA'].nunique()} únicas")
    actividad_rec = df.groupby('ACTIVIDAD_ECONOMICA')['VALOR_RECAUDADO'].sum().sort_values(ascending=False)
    actividad_pct = (actividad_rec / actividad_rec.sum() * 100).round(2)
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Actividades", f"{len(actividad_rec):,}")
    with col2:
        st.metric("Top Actividad", actividad_rec.index[0][:30] + "...")
    with col3:
        st.metric("% de la Top", f"{actividad_pct.iloc[0]:.1f}%")
    
    # Top 15 actividades
    top_15 = actividad_rec.head(15)
    top_15_pct = actividad_pct.head(15)
    
    fig = go.Figure(go.Bar(
        y=top_15.index,
        x=top_15.values / 1e6,
        orientation='h',
        marker=dict(
            color=top_15.values,
            colorscale='Greens',
            showscale=True,
            colorbar=dict(title="Millones $")
        ),
        text=[f"{pct}%" for pct in top_15_pct],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>$%{x:.1f}M<br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Top 15 Actividades Económicas por Recaudación</b>",
        xaxis_title="<b>Millones de Dólares ($)</b>",
        yaxis_title="",
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ No hay datos de actividad económica disponibles")

st.markdown("---")

# Sección 6: Tipo de Contribuyente
st.header("6. Análisis por Tipo de Contribuyente")

if 'TIPO_CONTRIBUYENTE' in df.columns and 'VALOR_RECAUDADO' in df.columns:
    tipo_contrib = df.groupby('TIPO_CONTRIBUYENTE').agg({
        'VALOR_RECAUDADO': ['sum', 'mean', 'count']
    }).reset_index()
    
    tipo_contrib.columns = ['TIPO_CONTRIBUYENTE', 'TOTAL_RECAUDADO', 'PROMEDIO_RECAUDADO', 'NUM_REGISTROS']
    tipo_contrib = tipo_contrib.sort_values('TOTAL_RECAUDADO', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras
        fig = px.bar(
            tipo_contrib,
            x='TIPO_CONTRIBUYENTE',
            y='TOTAL_RECAUDADO',
            title="Recaudación Total por Tipo de Contribuyente",
            labels={'TOTAL_RECAUDADO': 'Recaudación Total ($)', 'TIPO_CONTRIBUYENTE': 'Tipo'},
            color='TOTAL_RECAUDADO',
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Promedio por tipo
        fig = px.bar(
            tipo_contrib,
            x='TIPO_CONTRIBUYENTE',
            y='PROMEDIO_RECAUDADO',
            title="Recaudación Promedio por Tipo de Contribuyente",
            labels={'PROMEDIO_RECAUDADO': 'Promedio ($)', 'TIPO_CONTRIBUYENTE': 'Tipo'},
            color='PROMEDIO_RECAUDADO',
            color_continuous_scale='Purples'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar imagen guardada
    fig_path = Path(__file__).parent.parent / "Fig_8_Tipo_Contribuyente.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Análisis por Tipo de Contribuyente (del notebook)", use_container_width=True)

st.markdown("---")

# Sección 7: Análisis de Recaudación
st.header("7. Distribución de Valores de Recaudación")

if 'VALOR_RECAUDADO' in df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma
        fig = px.histogram(
            df,
            x='VALOR_RECAUDADO',
            nbins=50,
            title="Distribución de Valores de Recaudación",
            labels={'VALOR_RECAUDADO': 'Valor Recaudado ($)'},
            marginal="box"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot por año si está disponible
        if 'ANIO' in df.columns:
            fig = px.box(
                df,
                x='ANIO',
                y='VALOR_RECAUDADO',
                title="Distribución de Recaudación por Año",
                labels={'VALOR_RECAUDADO': 'Valor Recaudado ($)', 'ANIO': 'Año'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.info("""
**Nota:** Todas las visualizaciones son generadas automáticamente del dataset preprocesado.  
Las imágenes adicionales provienen de los análisis realizados en los notebooks de Jupyter.
""")
