"""
Validador Robusto de Datos
Sistema Balístico Forense MVP

Validación avanzada de entrada de datos con esquemas, sanitización y verificación de integridad
"""

import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from utils.logger import LoggerMixin

# Importar sistema de monitoreo de rendimiento
try:
    from core.performance_monitor import monitor_performance, OperationType
except ImportError:
    # Fallback si el módulo no está disponible
    def monitor_performance(operation_type):
        def decorator(func):
            return func
        return decorator
    
    class OperationType:
        DATA_VALIDATION = "data_validation"


class ValidationSeverity(Enum):
    """Niveles de severidad de validación"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataType(Enum):
    """Tipos de datos soportados"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"
    FILE_PATH = "file_path"
    IMAGE_PATH = "image_path"
    CASE_NUMBER = "case_number"
    INVESTIGATOR_NAME = "investigator_name"
    WEAPON_TYPE = "weapon_type"
    CALIBER = "caliber"
    EVIDENCE_TYPE = "evidence_type"


@dataclass
class ValidationRule:
    """Regla de validación"""
    field_name: str
    data_type: DataType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    sanitize: bool = True
    description: str = ""


@dataclass
class ValidationError:
    """Error de validación"""
    field_name: str
    error_message: str
    severity: ValidationSeverity
    value: Any = None
    expected_type: Optional[DataType] = None


@dataclass
class ValidationResult:
    """Resultado de validación"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    sanitized_data: Dict[str, Any] = field(default_factory=dict)
    validation_time: float = 0.0
    
    def has_errors(self) -> bool:
        """Verifica si hay errores críticos"""
        return any(error.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for error in self.errors)
    
    def has_warnings(self) -> bool:
        """Verifica si hay advertencias"""
        return len(self.warnings) > 0
    
    def get_error_summary(self) -> str:
        """Obtiene resumen de errores"""
        if not self.errors:
            return "No errors"
        
        error_counts = {}
        for error in self.errors:
            severity = error.severity.value
            error_counts[severity] = error_counts.get(severity, 0) + 1
        
        return f"Errors: {error_counts}"


class DataValidator(LoggerMixin):
    """Validador robusto de datos con esquemas y sanitización avanzada"""
    
    # Patrones de validación predefinidos
    PATTERNS = {
        DataType.EMAIL: r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        DataType.URL: r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$',
        DataType.CASE_NUMBER: r'^[A-Z0-9\-_]{3,20}$',
        DataType.INVESTIGATOR_NAME: r'^[A-Za-z\s\.\-]{2,50}$',
        DataType.WEAPON_TYPE: r'^[A-Za-z\s\-]{2,30}$',
        DataType.CALIBER: r'^[A-Za-z0-9\.\-\s]{1,20}$',
        DataType.EVIDENCE_TYPE: r'^(vaina|proyectil|arma|otro)$'
    }
    
    # Valores permitidos para campos específicos
    ALLOWED_VALUES = {
        DataType.EVIDENCE_TYPE: ['vaina', 'proyectil', 'arma', 'otro'],
        DataType.WEAPON_TYPE: ['pistola', 'revolver', 'rifle', 'escopeta', 'otro'],
        DataType.CALIBER: ['.22', '.38', '.45', '9mm', '.40', '.357', '.44', 'otro']
    }
    
    def __init__(self):
        super().__init__()
        self.schemas: Dict[str, List[ValidationRule]] = {}
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compila patrones regex para mejor rendimiento"""
        for data_type, pattern in self.PATTERNS.items():
            self._compiled_patterns[data_type] = re.compile(pattern, re.IGNORECASE)
    
    def register_schema(self, schema_name: str, rules: List[ValidationRule]):
        """Registra un esquema de validación"""
        self.schemas[schema_name] = rules
        self.logger.info(f"Schema '{schema_name}' registered with {len(rules)} rules")
    
    @monitor_performance(OperationType.DATA_VALIDATION)
    def validate_data(self, data: Dict[str, Any], schema_name: str) -> ValidationResult:
        """Valida datos contra un esquema registrado"""
        start_time = datetime.now()
        
        if schema_name not in self.schemas:
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    field_name="schema",
                    error_message=f"Schema '{schema_name}' not found",
                    severity=ValidationSeverity.CRITICAL
                )]
            )
        
        rules = self.schemas[schema_name]
        errors = []
        warnings = []
        sanitized_data = {}
        
        # Validar campos requeridos
        for rule in rules:
            field_value = data.get(rule.field_name)
            
            # Verificar campos requeridos
            if rule.required and (field_value is None or field_value == ""):
                errors.append(ValidationError(
                    field_name=rule.field_name,
                    error_message=f"Field '{rule.field_name}' is required",
                    severity=ValidationSeverity.ERROR,
                    expected_type=rule.data_type
                ))
                continue
            
            # Si el campo no es requerido y está vacío, continuar
            if not rule.required and (field_value is None or field_value == ""):
                sanitized_data[rule.field_name] = field_value
                continue
            
            # Validar y sanitizar el campo
            field_errors, field_warnings, sanitized_value = self._validate_field(
                rule, field_value
            )
            
            errors.extend(field_errors)
            warnings.extend(field_warnings)
            sanitized_data[rule.field_name] = sanitized_value
        
        # Verificar campos no definidos en el esquema
        for field_name, field_value in data.items():
            if not any(rule.field_name == field_name for rule in rules):
                warnings.append(ValidationError(
                    field_name=field_name,
                    error_message=f"Field '{field_name}' not defined in schema",
                    severity=ValidationSeverity.WARNING,
                    value=field_value
                ))
        
        validation_time = (datetime.now() - start_time).total_seconds()
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized_data,
            validation_time=validation_time
        )
        
        self.logger.info(f"Validation completed for schema '{schema_name}': "
                        f"{'PASSED' if result.is_valid else 'FAILED'} "
                        f"({len(errors)} errors, {len(warnings)} warnings)")
        
        return result
    
    def _validate_field(self, rule: ValidationRule, value: Any) -> Tuple[List[ValidationError], List[ValidationError], Any]:
        """Valida un campo individual"""
        errors = []
        warnings = []
        sanitized_value = value
        
        try:
            # Sanitizar si es necesario
            if rule.sanitize and isinstance(value, str):
                sanitized_value = self._sanitize_string(value)
            
            # Validar tipo de dato
            type_errors, converted_value = self._validate_data_type(rule, sanitized_value)
            errors.extend(type_errors)
            
            if type_errors:
                return errors, warnings, sanitized_value
            
            sanitized_value = converted_value
            
            # Validar longitud (para strings)
            if isinstance(sanitized_value, str):
                length_errors = self._validate_length(rule, sanitized_value)
                errors.extend(length_errors)
            
            # Validar rango (para números)
            if isinstance(sanitized_value, (int, float)):
                range_errors = self._validate_range(rule, sanitized_value)
                errors.extend(range_errors)
            
            # Validar patrón
            if rule.pattern or rule.data_type in self._compiled_patterns:
                pattern_errors = self._validate_pattern(rule, sanitized_value)
                errors.extend(pattern_errors)
            
            # Validar valores permitidos
            if rule.allowed_values or rule.data_type in self.ALLOWED_VALUES:
                allowed_errors = self._validate_allowed_values(rule, sanitized_value)
                errors.extend(allowed_errors)
            
            # Validador personalizado
            if rule.custom_validator:
                custom_errors = self._validate_custom(rule, sanitized_value)
                errors.extend(custom_errors)
                
        except Exception as e:
            errors.append(ValidationError(
                field_name=rule.field_name,
                error_message=f"Validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return errors, warnings, sanitized_value
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitiza strings de entrada"""
        if not isinstance(value, str):
            return value
        
        # Remover caracteres de control
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
        
        # Normalizar espacios
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Remover espacios al inicio y final
        sanitized = sanitized.strip()
        
        return sanitized
    
    def _validate_data_type(self, rule: ValidationRule, value: Any) -> Tuple[List[ValidationError], Any]:
        """Valida y convierte tipos de datos"""
        errors = []
        converted_value = value
        
        try:
            if rule.data_type == DataType.STRING:
                converted_value = str(value) if value is not None else ""
            
            elif rule.data_type == DataType.INTEGER:
                if isinstance(value, str) and value.strip() == "":
                    converted_value = None
                else:
                    converted_value = int(value)
            
            elif rule.data_type == DataType.FLOAT:
                if isinstance(value, str) and value.strip() == "":
                    converted_value = None
                else:
                    converted_value = float(value)
            
            elif rule.data_type == DataType.BOOLEAN:
                if isinstance(value, str):
                    converted_value = value.lower() in ['true', '1', 'yes', 'on', 'si']
                else:
                    converted_value = bool(value)
            
            elif rule.data_type == DataType.DATE:
                if isinstance(value, str):
                    converted_value = datetime.strptime(value, '%Y-%m-%d').date()
                elif isinstance(value, datetime):
                    converted_value = value.date()
                elif not isinstance(value, date):
                    raise ValueError("Invalid date format")
            
            elif rule.data_type == DataType.DATETIME:
                if isinstance(value, str):
                    # Intentar varios formatos
                    formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']
                    for fmt in formats:
                        try:
                            converted_value = datetime.strptime(value, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError("Invalid datetime format")
                elif not isinstance(value, datetime):
                    raise ValueError("Invalid datetime type")
            
            # Para tipos específicos del dominio, tratar como string
            elif rule.data_type in [DataType.EMAIL, DataType.URL, DataType.FILE_PATH, 
                                   DataType.IMAGE_PATH, DataType.CASE_NUMBER, 
                                   DataType.INVESTIGATOR_NAME, DataType.WEAPON_TYPE, 
                                   DataType.CALIBER, DataType.EVIDENCE_TYPE]:
                converted_value = str(value) if value is not None else ""
                
        except (ValueError, TypeError) as e:
            errors.append(ValidationError(
                field_name=rule.field_name,
                error_message=f"Invalid {rule.data_type.value}: {str(e)}",
                severity=ValidationSeverity.ERROR,
                value=value,
                expected_type=rule.data_type
            ))
        
        return errors, converted_value
    
    def _validate_length(self, rule: ValidationRule, value: str) -> List[ValidationError]:
        """Valida longitud de strings"""
        errors = []
        
        if rule.min_length is not None and len(value) < rule.min_length:
            errors.append(ValidationError(
                field_name=rule.field_name,
                error_message=f"Minimum length is {rule.min_length}, got {len(value)}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        if rule.max_length is not None and len(value) > rule.max_length:
            errors.append(ValidationError(
                field_name=rule.field_name,
                error_message=f"Maximum length is {rule.max_length}, got {len(value)}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return errors
    
    def _validate_range(self, rule: ValidationRule, value: Union[int, float]) -> List[ValidationError]:
        """Valida rango de números"""
        errors = []
        
        if rule.min_value is not None and value < rule.min_value:
            errors.append(ValidationError(
                field_name=rule.field_name,
                error_message=f"Minimum value is {rule.min_value}, got {value}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        if rule.max_value is not None and value > rule.max_value:
            errors.append(ValidationError(
                field_name=rule.field_name,
                error_message=f"Maximum value is {rule.max_value}, got {value}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return errors
    
    def _validate_pattern(self, rule: ValidationRule, value: str) -> List[ValidationError]:
        """Valida patrones regex"""
        errors = []
        
        # Usar patrón personalizado o patrón predefinido
        pattern = None
        if rule.pattern:
            pattern = re.compile(rule.pattern, re.IGNORECASE)
        elif rule.data_type in self._compiled_patterns:
            pattern = self._compiled_patterns[rule.data_type]
        
        if pattern and not pattern.match(str(value)):
            errors.append(ValidationError(
                field_name=rule.field_name,
                error_message=f"Value does not match required pattern for {rule.data_type.value}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return errors
    
    def _validate_allowed_values(self, rule: ValidationRule, value: Any) -> List[ValidationError]:
        """Valida valores permitidos"""
        errors = []
        
        # Usar valores permitidos personalizados o predefinidos
        allowed_values = rule.allowed_values
        if not allowed_values and rule.data_type in self.ALLOWED_VALUES:
            allowed_values = self.ALLOWED_VALUES[rule.data_type]
        
        if allowed_values and value not in allowed_values:
            errors.append(ValidationError(
                field_name=rule.field_name,
                error_message=f"Value must be one of: {', '.join(map(str, allowed_values))}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return errors
    
    def _validate_custom(self, rule: ValidationRule, value: Any) -> List[ValidationError]:
        """Ejecuta validador personalizado"""
        errors = []
        
        try:
            is_valid, error_message = rule.custom_validator(value)
            if not is_valid:
                errors.append(ValidationError(
                    field_name=rule.field_name,
                    error_message=error_message,
                    severity=ValidationSeverity.ERROR,
                    value=value
                ))
        except Exception as e:
            errors.append(ValidationError(
                field_name=rule.field_name,
                error_message=f"Custom validator error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return errors
    
    def create_ballistic_case_schema(self) -> List[ValidationRule]:
        """Crea esquema de validación para casos balísticos"""
        return [
            ValidationRule(
                field_name="case_number",
                data_type=DataType.CASE_NUMBER,
                required=True,
                min_length=3,
                max_length=20,
                description="Número único del caso"
            ),
            ValidationRule(
                field_name="investigator",
                data_type=DataType.INVESTIGATOR_NAME,
                required=True,
                min_length=2,
                max_length=50,
                description="Nombre del investigador"
            ),
            ValidationRule(
                field_name="weapon_type",
                data_type=DataType.WEAPON_TYPE,
                required=False,
                description="Tipo de arma"
            ),
            ValidationRule(
                field_name="weapon_model",
                data_type=DataType.STRING,
                required=False,
                max_length=50,
                description="Modelo del arma"
            ),
            ValidationRule(
                field_name="caliber",
                data_type=DataType.CALIBER,
                required=False,
                description="Calibre del arma"
            ),
            ValidationRule(
                field_name="description",
                data_type=DataType.STRING,
                required=False,
                max_length=1000,
                description="Descripción del caso"
            ),
            ValidationRule(
                field_name="date_created",
                data_type=DataType.DATETIME,
                required=True,
                description="Fecha de creación del caso"
            )
        ]
    
    def create_ballistic_image_schema(self) -> List[ValidationRule]:
        """Crea esquema de validación para imágenes balísticas"""
        return [
            ValidationRule(
                field_name="filename",
                data_type=DataType.STRING,
                required=True,
                min_length=1,
                max_length=255,
                description="Nombre del archivo"
            ),
            ValidationRule(
                field_name="file_path",
                data_type=DataType.FILE_PATH,
                required=True,
                description="Ruta del archivo"
            ),
            ValidationRule(
                field_name="evidence_type",
                data_type=DataType.EVIDENCE_TYPE,
                required=True,
                description="Tipo de evidencia"
            ),
            ValidationRule(
                field_name="width",
                data_type=DataType.INTEGER,
                required=False,
                min_value=1,
                max_value=10000,
                description="Ancho de la imagen"
            ),
            ValidationRule(
                field_name="height",
                data_type=DataType.INTEGER,
                required=False,
                min_value=1,
                max_value=10000,
                description="Alto de la imagen"
            ),
            ValidationRule(
                field_name="file_size",
                data_type=DataType.INTEGER,
                required=False,
                min_value=1,
                description="Tamaño del archivo en bytes"
            )
        ]


# Instancia global del validador
_global_validator = None

def get_data_validator() -> DataValidator:
    """Obtiene instancia global del validador de datos"""
    global _global_validator
    if _global_validator is None:
        _global_validator = DataValidator()
        
        # Registrar esquemas predefinidos
        _global_validator.register_schema(
            "ballistic_case", 
            _global_validator.create_ballistic_case_schema()
        )
        _global_validator.register_schema(
            "ballistic_image", 
            _global_validator.create_ballistic_image_schema()
        )
    
    return _global_validator