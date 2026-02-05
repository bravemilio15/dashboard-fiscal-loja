"""
Utilidad para iconos HTML con Material Icons
Similar a shadcn pero para Streamlit
"""

def icon(name, size=20, color="#000000"):
    """
    Genera un ícono Material Icons
    
    Args:
        name: Nombre del ícono de Material Icons
        size: Tamaño en px
        color: Color en hex
    
    Ejemplo:
        icon("check_circle", 24, "#2ecc71")
    """
    return f'<span class="material-icons" style="font-size:{size}px;color:{color};vertical-align:middle">{name}</span>'

def icon_text(icon_name, text, icon_size=20, icon_color="#000000"):
    """
    Genera ícono + texto
    
    Ejemplo:
        icon_text("analytics", "Panel de KPIs")
    """
    return f'{icon(icon_name, icon_size, icon_color)} {text}'

# CDN para Material Icons (agregar en el head del app)
MATERIAL_ICONS_CDN = """
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
"""

# Iconos comunes del proyecto
class Icons:
    # Generales
    CHECK = "check_circle"
    CLOSE = "cancel"
    WARNING = "warning"
    INFO = "info"
    ERROR = "error"
    
    # Métricas
    MONEY = "attach_money"
    TRENDING_UP = "trending_up"
    TARGET = "gps_fixed"
    CHART = "bar_chart"
    BAR_CHART = "bar_chart"
    PIE = "pie_chart"
    
    # Acciones
    SEARCH = "search"
    FILTER = "filter_alt"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    
    # Datos
    TABLE = "table_chart"
    DATABASE = "storage"
    ANALYTICS = "analytics"
    INSIGHTS = "insights"
    
    # Ubicación
    LOCATION = "location_on"
    MAP = "map"
    
    # Tiempo
    CALENDAR = "calendar_today"
    SCHEDULE = "schedule"
    
    # ML
    MODEL = "model_training"
    SCATTER = "scatter_plot"
    TREE = "account_tree"
    CLUSTER = "hub"
