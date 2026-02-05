"""
Utilidades compartidas para el dashboard
"""

from .data_loader import (
    load_dataset,
    validate_dataset,
    get_numeric_columns,
    get_categorical_columns
)

__all__ = [
    'load_dataset',
    'validate_dataset',
    'get_numeric_columns',
    'get_categorical_columns'
]
