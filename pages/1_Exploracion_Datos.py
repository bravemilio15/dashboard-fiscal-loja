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
            title="<b>Distribución: Contribuyentes que Tributan vs No Tributan</b>",
            hole=0.4,
            color_discrete_sequence=['#27AE60', '#E74C3C']
        )
        fig.update_traces(
            textposition='outside',
            textinfo='percent+label',
            textfont=dict(size=15, color='#000000', family='Arial Black'),
            marker=dict(line=dict(color='white', width=3))
        )
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=18, color='#2C3E50', family='Arial Black'),
            legend=dict(font=dict(size=14, color='#000000'))
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Gráfico de barras
        fig = px.bar(
            x=['Tributan (0)', 'No Tributan (1)'],
            y=[tributan, no_tributan],
            title="<b>Comparación de Contribuyentes</b>",
            labels={'x': 'Categoría', 'y': 'Cantidad'},
            color=['Tributan', 'No Tributan'],
            color_discrete_sequence=['#27AE60', '#E74C3C'],
            text=[tributan, no_tributan]
        )
        fig.update_traces(
            texttemplate='<b>%{text:,}</b>',
            textposition='outside',
            textfont=dict(size=16, color='#000000', family='Arial Black')
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=18, color='#2C3E50', family='Arial Black'),
            yaxis=dict(gridcolor='#E8E8E8', range=[0, max(tributan, no_tributan) * 1.15])
        )
        st.plotly_chart(fig, width='stretch')
    
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
            line=dict(color='#2980B9', width=4),
            marker=dict(size=14, line=dict(width=2, color='white'))
        ))
        
        # Agregar anotaciones en cada punto
        for i, row in recaudacion_anual.iterrows():
            fig.add_annotation(
                x=row['ANIO'],
                y=row['sum_millones'],
                text=f"<b>${row['sum_millones']:.1f}M</b>",
                showarrow=False,
                yshift=20,
                font=dict(size=13, color='#000000', family='Arial Black'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#2980B9',
                borderwidth=2,
                borderpad=4
            )
        
        fig.update_layout(
            title="<b>Recaudación Total por Año (Millones $)</b>",
            xaxis_title="<b>Año</b>",
            yaxis_title="<b>Recaudación (Millones $)</b>",
            height=400,
            hovermode='x unified',
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=18, color='#2C3E50', family='Arial Black'),
            xaxis=dict(tickformat='d', gridcolor='#E8E8E8'),
            yaxis=dict(gridcolor='#E8E8E8', range=[0, recaudacion_anual['sum_millones'].max() * 1.2])
        )
        
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Gráfico de barras con cantidad de registros
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=recaudacion_anual['ANIO'],
            y=recaudacion_anual['count'],
            marker=dict(
                color=recaudacion_anual['count'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(text="<b>Registros</b>", font=dict(size=14, color='#000000')),
                    tickfont=dict(size=12, color='#000000')
                ),
                line=dict(color='white', width=2)
            ),
            text=[f"<b>{val:,}</b>" for val in recaudacion_anual['count']],
            textposition='outside',
            textfont=dict(size=14, color='#000000', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>%{y:,} registros<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>Cantidad de Registros por Año</b>",
            xaxis_title="<b>Año</b>",
            yaxis_title="<b>Número de Registros</b>",
            height=400,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=18, color='#2C3E50', family='Arial Black'),
            xaxis=dict(tickformat='d', gridcolor='#E8E8E8'),
            yaxis=dict(gridcolor='#E8E8E8', range=[0, recaudacion_anual['count'].max() * 1.15])
        )
        st.plotly_chart(fig, width='stretch')
    
    # Tabla resumen
    st.markdown("### Resumen Anual")
    resumen_display = recaudacion_anual[['ANIO', 'sum_millones', 'count']].copy()
    resumen_display.columns = ['Año', 'Recaudación Total (Millones $)', 'Número de Registros']
    resumen_display['Recaudación Total (Millones $)'] = resumen_display['Recaudación Total (Millones $)'].round(2)
    st.dataframe(resumen_display, width='stretch', hide_index=True)

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
        top_10 = recaudacion_canton.head(10)
        # Colorear LOJA diferente
        colors = ['#E74C3C' if canton == 'LOJA' else '#3498DB' for canton in top_10['CANTON']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_10['CANTON'],
            x=top_10['TOTAL_RECAUDADO'] / 1e6,
            orientation='h',
            text=[f"<b>{pct}%</b>" for pct in top_10['PORCENTAJE']],
            textposition='outside',
            textfont=dict(size=14, color='#000000', family='Arial Black'),
            marker=dict(color=colors, line=dict(color='white', width=2)),
            hovertemplate='<b>%{y}</b><br>$%{x:.1f}M (%{text})<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>Top 10 Cantones por Recaudación Total</b>",
            xaxis_title="<b>Millones de Dólares ($)</b>",
            yaxis_title="",
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=18, color='#2C3E50', family='Arial Black'),
            xaxis=dict(
                gridcolor='#E8E8E8',
                range=[0, (top_10['TOTAL_RECAUDADO'].max() / 1e6) * 1.25]
            ),
            margin=dict(l=120, r=180, t=60, b=60)
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Gráfico de pastel para top 5
        top_5 = recaudacion_canton.head(5)
        colors = ['#E74C3C' if c == 'LOJA' else '#3498DB' for c in top_5['CANTON']]
        
        # Recalcular porcentajes correctamente basados en el top 5
        top_5_copy = top_5.copy()
        top_5_copy['PORCENTAJE'] = (top_5_copy['TOTAL_RECAUDADO'] / top_5_copy['TOTAL_RECAUDADO'].sum() * 100).round(1)
        
        fig = go.Figure(data=[go.Pie(
            labels=top_5_copy['CANTON'],
            values=top_5_copy['TOTAL_RECAUDADO'],
            hole=0.5,
            marker=dict(colors=colors, line=dict(color='white', width=3)),
            textinfo='percent',
            textfont=dict(size=16, color='#000000', family='Arial Black'),
            textposition='outside',
            insidetextorientation='radial',
            pull=[0.05 if c == 'LOJA' else 0 for c in top_5_copy['CANTON']],
            hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="<b>Distribución Top 5 Cantones</b>",
            height=500,
            template='plotly_white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=16, color='#2C3E50', family='Arial Black'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=13, color='#000000', family='Arial Black')
            ),
            margin=dict(l=100, r=100, t=100, b=120)
        )
        st.plotly_chart(fig, width='stretch')
    
    # Mostrar imagen guardada del notebook
    fig_path = Path(__file__).parent.parent / "Fig_7_Canton.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Análisis de Cantones (del notebook)", width='stretch')

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
            colorbar=dict(
                title=dict(text="<b>Millones $</b>", font=dict(size=14, color='#000000')),
                tickfont=dict(size=12, color='#000000')
            ),
            line=dict(color='white', width=1)
        ),
        text=[f"<b>{pct}%</b>" for pct in top_15_pct],
        textposition='outside',
        textfont=dict(size=13, color='#000000', family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>$%{x:.1f}M<br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Top 15 Actividades Económicas por Recaudación</b>",
        xaxis_title="<b>Millones de Dólares ($)</b>",
        yaxis_title="",
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=13, color='#000000', family='Arial'),
        title_font=dict(size=18, color='#2C3E50', family='Arial Black'),
        xaxis=dict(gridcolor='#E8E8E8'),
        margin=dict(l=200, r=80)
    )
    st.plotly_chart(fig, width='stretch')
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
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tipo_contrib['TIPO_CONTRIBUYENTE'],
            y=tipo_contrib['TOTAL_RECAUDADO'] / 1e6,
            marker=dict(
                color=['#D35400', '#E67E22', '#F39C12'],
                line=dict(color='white', width=2)
            ),
            text=[f"<b>${val/1e6:.1f}M</b>" for val in tipo_contrib['TOTAL_RECAUDADO']],
            textposition='outside',
            textfont=dict(size=15, color='#000000', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>$%{y:.1f}M<extra></extra>'
        ))
        fig.update_layout(
            title="<b>Recaudación Total por Tipo de Contribuyente</b>",
            xaxis_title="<b>Tipo de Contribuyente</b>",
            yaxis_title="<b>Millones de Dólares ($)</b>",
            height=400,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=16, color='#2C3E50', family='Arial Black'),
            yaxis=dict(gridcolor='#E8E8E8', range=[0, tipo_contrib['TOTAL_RECAUDADO'].max() / 1e6 * 1.15]),
            xaxis=dict(tickangle=-15)
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Promedio por tipo
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tipo_contrib['TIPO_CONTRIBUYENTE'],
            y=tipo_contrib['PROMEDIO_RECAUDADO'],
            marker=dict(
                color=['#6C3483', '#8E44AD', '#9B59B6'],
                line=dict(color='white', width=2)
            ),
            text=[f"<b>${val:,.0f}</b>" for val in tipo_contrib['PROMEDIO_RECAUDADO']],
            textposition='outside',
            textfont=dict(size=15, color='#000000', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>$%{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(
            title="<b>Recaudación Promedio por Tipo de Contribuyente</b>",
            xaxis_title="<b>Tipo de Contribuyente</b>",
            yaxis_title="<b>Promedio ($)</b>",
            height=400,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=16, color='#2C3E50', family='Arial Black'),
            yaxis=dict(gridcolor='#E8E8E8', range=[0, tipo_contrib['PROMEDIO_RECAUDADO'].max() * 1.15]),
            xaxis=dict(tickangle=-15)
        )
        st.plotly_chart(fig, width='stretch')
    
    # Mostrar imagen guardada
    fig_path = Path(__file__).parent.parent / "Fig_8_Tipo_Contribuyente.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Análisis por Tipo de Contribuyente (del notebook)", width='stretch')

st.markdown("---")

# Sección 7: Análisis de Recaudación
st.header("7. Distribución de Valores de Recaudación")

if 'VALOR_RECAUDADO' in df.columns:
    # Métricas resumen
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Promedio General", f"${df['VALOR_RECAUDADO'].mean():,.0f}")
    with col2:
        st.metric("Mediana", f"${df['VALOR_RECAUDADO'].median():,.0f}")
    with col3:
        st.metric("Valor Máximo", f"${df['VALOR_RECAUDADO'].max():,.0f}")
    with col4:
        contribuyentes_pagan = (df['VALOR_RECAUDADO'] > 0).sum()
        st.metric("Contribuyentes que Pagan", f"{contribuyentes_pagan:,}")
    
    st.markdown("")
    
    # Crear rangos de recaudación para mejor comprensión
    df_rangos = df[df['VALOR_RECAUDADO'] > 0].copy()
    
    def clasificar_monto(valor):
        if valor <= 100:
            return "Micro: $0-$100"
        elif valor <= 1000:
            return "Pequeño: $100-$1K"
        elif valor <= 10000:
            return "Mediano: $1K-$10K"
        elif valor <= 100000:
            return "Grande: $10K-$100K"
        else:
            return "Elite: >$100K"
    
    df_rangos['RANGO'] = df_rangos['VALOR_RECAUDADO'].apply(clasificar_monto)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras por rangos
        rangos_count = df_rangos['RANGO'].value_counts().reindex([
            "Micro: $0-$100", 
            "Pequeño: $100-$1K", 
            "Mediano: $1K-$10K", 
            "Grande: $10K-$100K", 
            "Elite: >$100K"
        ], fill_value=0)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rangos_count.index,
            y=rangos_count.values,
            marker=dict(
                color=['#27AE60', '#3498DB', '#F39C12', '#E67E22', '#E74C3C'],
                line=dict(color='white', width=2)
            ),
            text=[f"<b>{val:,}</b>" for val in rangos_count.values],
            textposition='outside',
            textfont=dict(size=14, color='#000000', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>%{y:,} contribuyentes<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>Distribución de Contribuyentes<br>por Rango de Recaudación</b>",
            xaxis_title="<b>Rango de Monto</b>",
            yaxis_title="<b>Número de Contribuyentes</b>",
            height=450,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=13, color='#000000', family='Arial'),
            title_font=dict(size=16, color='#2C3E50', family='Arial Black'),
            xaxis=dict(tickangle=-20, gridcolor='#E8E8E8'),
            yaxis=dict(gridcolor='#E8E8E8', range=[0, rangos_count.max() * 1.15]),
            margin=dict(t=100, b=100, l=80, r=80)
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Gráfico de pastel con % de recaudación por rango
        rangos_suma = df_rangos.groupby('RANGO')['VALOR_RECAUDADO'].sum().reindex([
            "Micro: $0-$100", 
            "Pequeño: $100-$1K", 
            "Mediano: $1K-$10K", 
            "Grande: $10K-$100K", 
            "Elite: >$100K"
        ], fill_value=0)
        
        fig = go.Figure(data=[go.Pie(
            labels=rangos_suma.index,
            values=rangos_suma.values,
            hole=0.5,
            marker=dict(
                colors=['#27AE60', '#3498DB', '#F39C12', '#E67E22', '#E74C3C'],
                line=dict(color='white', width=3)
            ),
            textinfo='percent+label',
            textfont=dict(size=12, color='#000000', family='Arial Black'),
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="<b>% del Monto Total<br>por Rango de Recaudación</b>",
            height=450,
            template='plotly_white',
            paper_bgcolor='white',
            font=dict(size=13, color='#000000', family='Arial'),
            title_font=dict(size=16, color='#2C3E50', family='Arial Black'),
            showlegend=False,
            margin=dict(l=20, r=20, t=100, b=80)
        )
        st.plotly_chart(fig, width='stretch')
    
    # Análisis por año (si está disponible)
    if 'ANIO' in df.columns:
        st.markdown("### 📈 Evolución de Rangos por Año")
        
        # Crear tabla pivote
        df_rangos_anio = df_rangos.groupby(['ANIO', 'RANGO'])['VALOR_RECAUDADO'].sum().reset_index()
        
        fig = px.bar(
            df_rangos_anio,
            x='ANIO',
            y='VALOR_RECAUDADO',
            color='RANGO',
            title="<b>Recaudación Total por Rango a lo Largo del Tiempo</b>",
            labels={'VALOR_RECAUDADO': 'Recaudación Total ($)', 'ANIO': 'Año'},
            color_discrete_map={
                "Micro: $0-$100": '#27AE60',
                "Pequeño: $100-$1K": '#3498DB',
                "Mediano: $1K-$10K": '#F39C12',
                "Grande: $10K-$100K": '#E67E22',
                "Elite: >$100K": '#E74C3C'
            },
            barmode='stack'
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14, color='#000000', family='Arial'),
            title_font=dict(size=16, color='#2C3E50', family='Arial Black'),
            xaxis=dict(tickformat='d', gridcolor='#E8E8E8'),
            yaxis=dict(gridcolor='#E8E8E8'),
            legend=dict(title="<b>Rango de Recaudación</b>", font=dict(size=12, color='#000000'))
        )
        st.plotly_chart(fig, width='stretch')

# Footer
st.markdown("---")
st.info("""
**Nota:** Todas las visualizaciones son generadas automáticamente del dataset preprocesado.  
Las imágenes adicionales provienen de los análisis realizados en los notebooks de Jupyter.
""")
