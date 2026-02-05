"""
Funciones de utilidad para carga y validación de datos
"""

import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, List


def load_dataset(file_path: str = None, uploaded_file=None) -> Optional[pd.DataFrame]:
    """
    Carga un dataset desde un archivo CSV o Excel
    
    Args:
        file_path: Ruta del archivo en el sistema
        uploaded_file: Archivo subido mediante st.file_uploader
        
    Returns:
        DataFrame de pandas o None si hay error
    """
    try:
        if uploaded_file is not None:
            # Cargar desde archivo subido
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error(f"Formato de archivo no soportado: {file_extension}")
                return None
                
        elif file_path is not None:
            # Cargar desde ruta del sistema
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                st.error(f"Formato de archivo no soportado: {file_extension}")
                return None
        else:
            st.error("Debe proporcionar un archivo o una ruta")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error al cargar el dataset: {str(e)}")
        return None


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Valida la estructura del dataset y retorna información básica
    
    Args:
        df: DataFrame de pandas
        
    Returns:
        Diccionario con información de validación
    """
    validation_info = {
        'valid': True,
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'warnings': []
    }
    
    # Validaciones
    if validation_info['num_rows'] == 0:
        validation_info['valid'] = False
        validation_info['warnings'].append("El dataset está vacío")
    
    if validation_info['missing_values'] > 0:
        validation_info['warnings'].append(
            f"Se encontraron {validation_info['missing_values']} valores faltantes"
        )
    
    if validation_info['duplicate_rows'] > 0:
        validation_info['warnings'].append(
            f"Se encontraron {validation_info['duplicate_rows']} filas duplicadas"
        )
    
    return validation_info


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Obtiene las columnas numéricas del dataset
    
    Args:
        df: DataFrame de pandas
        
    Returns:
        Lista de nombres de columnas numéricas
    """
    return df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Obtiene las columnas categóricas del dataset
    
    Args:
        df: DataFrame de pandas
        
    Returns:
        Lista de nombres de columnas categóricas
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()
