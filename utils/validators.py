"""
Validadores y Utilidades del Sistema
Sistema Balístico Forense MVP

Funciones de validación, sanitización y utilidades generales
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import mimetypes
import hashlib
from datetime import datetime

from utils.logger import LoggerMixin

class SystemValidator(LoggerMixin):
    """Validador del sistema para entradas y archivos"""
    
    # Extensiones de imagen permitidas
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    # Tipos MIME permitidos
    ALLOWED_MIME_TYPES = {
        'image/jpeg', 'image/png', 'image/tiff', 'image/bmp'
    }
    
    # Tamaño máximo de archivo (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    # Patrones de validación
    CASE_NUMBER_PATTERN = re.compile(r'^[A-Z0-9\-_]{3,20}$')
    INVESTIGATOR_PATTERN = re.compile(r'^[A-Za-z\s\.\-]{2,50}$')
    WEAPON_MODEL_PATTERN = re.compile(r'^[A-Za-z0-9\s\.\-/]{1,50}$')
    CALIBER_PATTERN = re.compile(r'^[A-Za-z0-9\.\-\s]{1,20}$')
    
    def __init__(self):
        pass
    
    def validate_image_file(self, file_path: str) -> Tuple[bool, str]:
        """Valida si un archivo es una imagen válida"""
        try:
            path = Path(file_path)
            
            # Verificar que el archivo existe
            if not path.exists():
                return False, f"Archivo no encontrado: {file_path}"
            
            # Verificar que es un archivo (no directorio)
            if not path.is_file():
                return False, f"La ruta no es un archivo: {file_path}"
            
            # Verificar extensión
            extension = path.suffix.lower()
            if extension not in self.ALLOWED_IMAGE_EXTENSIONS:
                return False, f"Extensión no permitida: {extension}. Permitidas: {', '.join(self.ALLOWED_IMAGE_EXTENSIONS)}"
            
            # Verificar tamaño
            file_size = path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                return False, f"Archivo demasiado grande: {file_size / (1024*1024):.1f}MB. Máximo: {self.MAX_FILE_SIZE / (1024*1024)}MB"
            
            # Verificar tipo MIME
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type not in self.ALLOWED_MIME_TYPES:
                return False, f"Tipo MIME no permitido: {mime_type}"
            
            # Verificar que no esté vacío
            if file_size == 0:
                return False, "El archivo está vacío"
            
            return True, "Archivo válido"
            
        except Exception as e:
            return False, f"Error validando archivo: {e}"
    
    def validate_case_number(self, case_number: str) -> Tuple[bool, str]:
        """Valida número de caso"""
        if not case_number:
            return False, "Número de caso requerido"
        
        case_number = case_number.strip().upper()
        
        if not self.CASE_NUMBER_PATTERN.match(case_number):
            return False, "Número de caso debe tener 3-20 caracteres alfanuméricos, guiones o guiones bajos"
        
        return True, "Número de caso válido"
    
    def validate_investigator_name(self, name: str) -> Tuple[bool, str]:
        """Valida nombre del investigador"""
        if not name:
            return False, "Nombre del investigador requerido"
        
        name = name.strip()
        
        if len(name) < 2:
            return False, "Nombre debe tener al menos 2 caracteres"
        
        if not self.INVESTIGATOR_PATTERN.match(name):
            return False, "Nombre contiene caracteres no válidos"
        
        return True, "Nombre válido"
    
    def validate_evidence_type(self, evidence_type: str) -> Tuple[bool, str]:
        """Valida tipo de evidencia"""
        valid_types = {'vaina', 'proyectil', 'casquillo', 'bala'}
        
        if not evidence_type:
            return False, "Tipo de evidencia requerido"
        
        evidence_type = evidence_type.lower().strip()
        
        if evidence_type not in valid_types:
            return False, f"Tipo de evidencia debe ser uno de: {', '.join(valid_types)}"
        
        return True, "Tipo de evidencia válido"
    
    def validate_weapon_info(self, weapon_type: str = "", weapon_model: str = "", 
                           caliber: str = "") -> Tuple[bool, str]:
        """Valida información del arma"""
        errors = []
        
        if weapon_type and len(weapon_type.strip()) > 50:
            errors.append("Tipo de arma demasiado largo (máximo 50 caracteres)")
        
        if weapon_model:
            weapon_model = weapon_model.strip()
            if not self.WEAPON_MODEL_PATTERN.match(weapon_model):
                errors.append("Modelo de arma contiene caracteres no válidos")
        
        if caliber:
            caliber = caliber.strip()
            if not self.CALIBER_PATTERN.match(caliber):
                errors.append("Calibre contiene caracteres no válidos")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, "Información de arma válida"
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitiza nombre de archivo para evitar problemas de seguridad"""
        # Remover caracteres peligrosos
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remover espacios múltiples y caracteres de control
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        # Limitar longitud
        if len(sanitized) > 100:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:100-len(ext)] + ext
        
        # Asegurar que no esté vacío
        if not sanitized or sanitized == '.':
            sanitized = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return sanitized
    
    def validate_directory_path(self, dir_path: str, create_if_missing: bool = False) -> Tuple[bool, str]:
        """Valida ruta de directorio"""
        try:
            path = Path(dir_path)
            
            # Verificar caracteres peligrosos
            if any(char in str(path) for char in ['<', '>', ':', '"', '|', '?', '*']):
                return False, "Ruta contiene caracteres no válidos"
            
            # Verificar que no sea una ruta relativa peligrosa
            if '..' in path.parts:
                return False, "Ruta no puede contener '..' por seguridad"
            
            if path.exists():
                if not path.is_dir():
                    return False, "La ruta existe pero no es un directorio"
                
                # Verificar permisos de escritura
                if not os.access(str(path), os.W_OK):
                    return False, "Sin permisos de escritura en el directorio"
            
            elif create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return False, f"No se pudo crear directorio: {e}"
            else:
                return False, "Directorio no existe"
            
            return True, "Directorio válido"
            
        except Exception as e:
            return False, f"Error validando directorio: {e}"
    
    def validate_batch_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """Valida múltiples imágenes en lote"""
        results = {
            'valid_files': [],
            'invalid_files': [],
            'total_size': 0,
            'errors': []
        }
        
        for image_path in image_paths:
            is_valid, message = self.validate_image_file(image_path)
            
            if is_valid:
                results['valid_files'].append(image_path)
                try:
                    results['total_size'] += Path(image_path).stat().st_size
                except:
                    pass
            else:
                results['invalid_files'].append({
                    'path': image_path,
                    'error': message
                })
                results['errors'].append(f"{Path(image_path).name}: {message}")
        
        return results

class SecurityUtils(LoggerMixin):
    """Utilidades de seguridad"""
    
    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> Optional[str]:
        """Calcula hash de un archivo"""
        try:
            hash_func = getattr(hashlib, algorithm)()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            return None
    
    @staticmethod
    def is_safe_path(path: str, base_path: str) -> bool:
        """Verifica que una ruta esté dentro del directorio base (previene path traversal)"""
        try:
            base = Path(base_path).resolve()
            target = Path(path).resolve()
            
            # Verificar que target esté dentro de base
            return str(target).startswith(str(base))
            
        except Exception:
            return False
    
    @staticmethod
    def sanitize_input(input_str: str, max_length: int = 1000) -> str:
        """Sanitiza entrada de usuario"""
        if not input_str:
            return ""
        
        # Remover caracteres de control
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', input_str)
        
        # Limitar longitud
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()

class FileUtils(LoggerMixin):
    """Utilidades para manejo de archivos"""
    
    @staticmethod
    def get_unique_filename(directory: str, base_name: str, extension: str = "") -> str:
        """Genera nombre de archivo único en un directorio"""
        counter = 1
        original_name = f"{base_name}{extension}"
        file_path = Path(directory) / original_name
        
        while file_path.exists():
            new_name = f"{base_name}_{counter}{extension}"
            file_path = Path(directory) / new_name
            counter += 1
        
        return str(file_path)
    
    @staticmethod
    def copy_file_safe(source: str, destination: str) -> Tuple[bool, str]:
        """Copia archivo de forma segura"""
        try:
            import shutil
            
            source_path = Path(source)
            dest_path = Path(destination)
            
            # Verificar que el archivo fuente existe
            if not source_path.exists():
                return False, "Archivo fuente no existe"
            
            # Crear directorio destino si no existe
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copiar archivo
            shutil.copy2(source, destination)
            
            return True, "Archivo copiado exitosamente"
            
        except Exception as e:
            return False, f"Error copiando archivo: {e}"
    
    @staticmethod
    def get_directory_size(directory: str) -> int:
        """Calcula tamaño total de un directorio"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except (OSError, IOError):
                        continue
        except Exception:
            pass
        
        return total_size
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Formatea tamaño de archivo en formato legible"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"