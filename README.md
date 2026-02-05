# Dashboard Fiscal Loja - Análisis de Recaudación 2020-2024

Dashboard interactivo para análisis de datos fiscales de la provincia de Loja, Ecuador. Proyecto de Data Mining con visualizaciones interactivas y modelos de Machine Learning.

## 🚀 Demo en Vivo

🔗 [Ver Dashboard](https://tu-usuario.streamlit.app)

## 📊 Características

- **Panel de KPIs**: Indicadores clave de recaudación fiscal
- **Exploración de Datos**: Análisis temporal, geográfico y sectorial
- **Validación de Hipótesis**: Concentración geográfica y Principio de Pareto
- **Modelos ML**: 
  - Isolation Forest (detección de anomalías)
  - K-Means (segmentación en 7 clústeres)
  - Árbol de Decisión (predicción de tributación)
  - Holt-Winters (proyección 2025)

## 🛠️ Tecnologías

- Python 3.8+
- Streamlit
- Plotly
- Pandas
- NumPy

## 📦 Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar dashboard
streamlit run dashboard_app.py
```

## 📁 Estructura del Proyecto

```
├── dashboard_app.py          # Dashboard principal
├── pages/
│   ├── 1_Exploracion_Datos.py
│   ├── 2_Hipotesis.py
│   └── 3_Modelos_ML.py
├── utils/
│   ├── icons.py              # Sistema de iconos Material
│   └── data_loader.py
├── Dataset_Loja_Preprocesado.csv
├── requirements.txt
└── README.md
```

## 📈 Datos

Dataset de recaudación fiscal del SRI (Servicio de Rentas Internas) de Ecuador:
- **Período**: 2020-2024
- **Registros**: 167,787
- **Región**: Provincia de Loja

## 🎓 Autor

Proyecto final - Data Mining
Universidad Nacional de Loja

## 📄 Licencia

MIT License
