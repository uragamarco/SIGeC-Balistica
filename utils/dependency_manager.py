#!/usr/bin/env python3
"""
SEACABAr - Sistema de Gestión de Dependencias
============================================

Este módulo proporciona un sistema centralizado para gestionar dependencias,
incluyendo validación automática, fallbacks robustos y manejo de dependencias opcionales.

Autor: SEACABAr Team
Versión: 1.0.0
"""

import sys
import subprocess
import importlib
import importlib.util
try:
    from importlib.metadata import version as get_version, PackageNotFoundError
except ImportError:
    # Fallback para Python < 3.8
    from importlib_metadata import version as get_version, PackageNotFoundError
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import warnings
import logging
from dataclasses import dataclass
from enum import Enum

# Configurar logging
logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Tipos de dependencias"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    DEVELOPMENT = "development"


class DependencyStatus(Enum):
    """Estados de las dependencias"""
    AVAILABLE = "available"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    IMPORT_ERROR = "import_error"


@dataclass
class DependencyInfo:
    """Información de una dependencia"""
    name: str
    import_name: str
    version_required: Optional[str] = None
    version_installed: Optional[str] = None
    dependency_type: DependencyType = DependencyType.REQUIRED
    fallback_available: bool = False
    fallback_function: Optional[Callable] = None
    status: DependencyStatus = DependencyStatus.MISSING
    error_message: Optional[str] = None


class DependencyManager:
    """
    Gestor centralizado de dependencias para SEACABAr
    
    Características:
    - Validación automática de dependencias
    - Fallbacks robustos para dependencias opcionales
    - Instalación automática de dependencias faltantes
    - Reportes detallados de estado
    """
    
    def __init__(self):
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.fallbacks: Dict[str, Any] = {}
        self._initialize_dependencies()
    
    def _initialize_dependencies(self):
        """Inicializa la lista de dependencias conocidas"""
        
        # Dependencias principales (requeridas)
        required_deps = [
            ("opencv-python", "cv2", "4.8.0.74"),
            ("numpy", "numpy", "1.24.3"),
            ("scipy", "scipy", "1.11.3"),
            ("scikit-image", "skimage", "0.20.0"),
            ("Pillow", "PIL", "10.0.1"),
            ("PyQt5", "PyQt5", "5.15.9"),
            ("faiss-cpu", "faiss", "1.7.4"),
            ("scikit-learn", "sklearn", "1.3.0"),
            ("pandas", "pandas", "2.0.3"),
            ("loguru", "loguru", "0.7.0"),
            ("pyyaml", "yaml", "6.0.1"),
        ]
        
        for pkg_name, import_name, version in required_deps:
            self.dependencies[pkg_name] = DependencyInfo(
                name=pkg_name,
                import_name=import_name,
                version_required=version,
                dependency_type=DependencyType.REQUIRED
            )
        
        # Dependencias opcionales
        optional_deps = [
            ("Flask", "flask", "2.3.3"),
            ("gunicorn", "gunicorn", "21.2.0"),
            ("flask-cors", "flask_cors", "4.0.0"),
            ("torch", "torch", "2.0.1"),
            ("torchvision", "torchvision", "0.15.2"),
            ("tensorflow", "tensorflow", "2.13.0"),
            ("rawpy", "rawpy", "0.18.1"),
            ("tifffile", "tifffile", "2023.7.10"),
        ]
        
        for pkg_name, import_name, version in optional_deps:
            self.dependencies[pkg_name] = DependencyInfo(
                name=pkg_name,
                import_name=import_name,
                version_required=version,
                dependency_type=DependencyType.OPTIONAL,
                fallback_available=True
            )
        
        # Dependencias de desarrollo
        dev_deps = [
            ("pytest", "pytest", "7.4.0"),
            ("pytest-qt", "pytestqt", "4.2.0"),
            ("pytest-cov", "pytest_cov", "4.1.0"),
            ("pyinstaller", "PyInstaller", "5.13.0"),
        ]
        
        for pkg_name, import_name, version in dev_deps:
            self.dependencies[pkg_name] = DependencyInfo(
                name=pkg_name,
                import_name=import_name,
                version_required=version,
                dependency_type=DependencyType.DEVELOPMENT
            )
    
    def check_dependency(self, package_name: str) -> DependencyInfo:
        """
        Verifica el estado de una dependencia específica
        
        Args:
            package_name: Nombre del paquete a verificar
            
        Returns:
            DependencyInfo con el estado actualizado
        """
        if package_name not in self.dependencies:
            return DependencyInfo(
                name=package_name,
                import_name=package_name,
                status=DependencyStatus.MISSING,
                error_message="Dependencia no registrada"
            )
        
        dep_info = self.dependencies[package_name]
        
        try:
            # Intentar importar el módulo
            module = importlib.import_module(dep_info.import_name)
            dep_info.status = DependencyStatus.AVAILABLE
            
            # Verificar versión si está especificada
            if dep_info.version_required:
                try:
                    installed_version = get_version(package_name)
                    dep_info.version_installed = installed_version
                    
                    if not self._is_version_compatible(installed_version, dep_info.version_required):
                        dep_info.status = DependencyStatus.VERSION_MISMATCH
                        dep_info.error_message = f"Versión instalada: {installed_version}, requerida: {dep_info.version_required}"
                
                except PackageNotFoundError:
                    dep_info.error_message = "No se pudo determinar la versión instalada"
            
        except ImportError as e:
            dep_info.status = DependencyStatus.IMPORT_ERROR
            dep_info.error_message = str(e)
        
        return dep_info
    
    def check_all_dependencies(self) -> Dict[str, DependencyInfo]:
        """
        Verifica todas las dependencias registradas
        
        Returns:
            Diccionario con el estado de todas las dependencias
        """
        results = {}
        for package_name in self.dependencies:
            results[package_name] = self.check_dependency(package_name)
        
        return results
    
    def get_missing_dependencies(self, include_optional: bool = False) -> List[str]:
        """
        Obtiene lista de dependencias faltantes
        
        Args:
            include_optional: Si incluir dependencias opcionales
            
        Returns:
            Lista de nombres de paquetes faltantes
        """
        missing = []
        results = self.check_all_dependencies()
        
        for package_name, dep_info in results.items():
            if dep_info.status in [DependencyStatus.MISSING, DependencyStatus.IMPORT_ERROR]:
                if dep_info.dependency_type == DependencyType.REQUIRED:
                    missing.append(package_name)
                elif include_optional and dep_info.dependency_type == DependencyType.OPTIONAL:
                    missing.append(package_name)
        
        return missing
    
    def install_dependency(self, package_name: str, version: Optional[str] = None) -> bool:
        """
        Instala una dependencia usando pip
        
        Args:
            package_name: Nombre del paquete
            version: Versión específica (opcional)
            
        Returns:
            True si la instalación fue exitosa
        """
        try:
            if version:
                package_spec = f"{package_name}=={version}"
            else:
                package_spec = package_name
            
            logger.info(f"Instalando {package_spec}...")
            
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_spec
            ], capture_output=True, text=True, check=True)
            
            logger.info(f"✅ {package_name} instalado correctamente")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error instalando {package_name}: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"❌ Error inesperado instalando {package_name}: {e}")
            return False
    
    def install_missing_dependencies(self, include_optional: bool = False) -> Tuple[List[str], List[str]]:
        """
        Instala todas las dependencias faltantes
        
        Args:
            include_optional: Si instalar dependencias opcionales
            
        Returns:
            Tupla con (instaladas_exitosamente, fallos)
        """
        missing = self.get_missing_dependencies(include_optional)
        installed = []
        failed = []
        
        for package_name in missing:
            dep_info = self.dependencies[package_name]
            if self.install_dependency(package_name, dep_info.version_required):
                installed.append(package_name)
            else:
                failed.append(package_name)
        
        return installed, failed
    
    def get_fallback(self, package_name: str) -> Any:
        """
        Obtiene el fallback para una dependencia opcional
        
        Args:
            package_name: Nombre del paquete
            
        Returns:
            Objeto fallback o None
        """
        if package_name in self.fallbacks:
            return self.fallbacks[package_name]
        
        # Fallbacks predefinidos
        fallback_implementations = {
            "torch": self._torch_fallback,
            "tensorflow": self._tensorflow_fallback,
            "Flask": self._flask_fallback,
            "rawpy": self._rawpy_fallback,
        }
        
        if package_name in fallback_implementations:
            fallback = fallback_implementations[package_name]()
            self.fallbacks[package_name] = fallback
            return fallback
        
        return None
    
    def safe_import(self, package_name: str, fallback_name: Optional[str] = None):
        """
        Importa un paquete de forma segura con fallback
        
        Args:
            package_name: Nombre del paquete a importar
            fallback_name: Nombre del fallback (opcional)
            
        Returns:
            Módulo importado o fallback
        """
        dep_info = self.check_dependency(package_name)
        
        if dep_info.status == DependencyStatus.AVAILABLE:
            return importlib.import_module(dep_info.import_name)
        
        # Si es una dependencia requerida, lanzar error
        if dep_info.dependency_type == DependencyType.REQUIRED:
            raise ImportError(f"Dependencia requerida '{package_name}' no disponible: {dep_info.error_message}")
        
        # Para dependencias opcionales, usar fallback
        fallback = self.get_fallback(fallback_name or package_name)
        if fallback:
            warnings.warn(f"Usando fallback para '{package_name}': funcionalidad limitada")
            return fallback
        
        warnings.warn(f"Dependencia opcional '{package_name}' no disponible y sin fallback")
        return None
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Genera un reporte completo del estado de dependencias
        
        Returns:
            Diccionario con el reporte
        """
        results = self.check_all_dependencies()
        
        report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "python_version": sys.version,
            "total_dependencies": len(results),
            "by_status": {
                "available": 0,
                "missing": 0,
                "version_mismatch": 0,
                "import_error": 0
            },
            "by_type": {
                "required": 0,
                "optional": 0,
                "development": 0
            },
            "details": {}
        }
        
        for package_name, dep_info in results.items():
            # Contar por estado
            report["by_status"][dep_info.status.value] += 1
            
            # Contar por tipo
            report["by_type"][dep_info.dependency_type.value] += 1
            
            # Detalles
            report["details"][package_name] = {
                "status": dep_info.status.value,
                "type": dep_info.dependency_type.value,
                "version_required": dep_info.version_required,
                "version_installed": dep_info.version_installed,
                "error_message": dep_info.error_message,
                "fallback_available": dep_info.fallback_available
            }
        
        return report
    
    def _is_version_compatible(self, installed: str, required: str) -> bool:
        """Verifica si la versión instalada es compatible con la requerida"""
        try:
            from packaging import version
            return version.parse(installed) >= version.parse(required)
        except ImportError:
            # Fallback simple si packaging no está disponible
            return installed == required
    
    # Implementaciones de fallback
    def _torch_fallback(self):
        """Fallback para PyTorch"""
        class TorchFallback:
            def __init__(self):
                self.available = False
                logger.warning("PyTorch no disponible - usando fallback con funcionalidad limitada")
            
            def tensor(self, data):
                import numpy as np
                return np.array(data)
            
            def load(self, path):
                raise NotImplementedError("PyTorch no disponible - no se pueden cargar modelos")
        
        return TorchFallback()
    
    def _tensorflow_fallback(self):
        """Fallback para TensorFlow"""
        class TensorFlowFallback:
            def __init__(self):
                self.available = False
                logger.warning("TensorFlow no disponible - usando fallback con funcionalidad limitada")
            
            def constant(self, data):
                import numpy as np
                return np.array(data)
        
        return TensorFlowFallback()
    
    def _flask_fallback(self):
        """Fallback para Flask"""
        class FlaskFallback:
            def __init__(self):
                self.available = False
                logger.warning("Flask no disponible - funcionalidad web deshabilitada")
            
            def Flask(self, name):
                raise NotImplementedError("Flask no disponible - funcionalidad web deshabilitada")
        
        return FlaskFallback()
    
    def _rawpy_fallback(self):
        """Fallback para rawpy"""
        class RawpyFallback:
            def __init__(self):
                self.available = False
                logger.warning("rawpy no disponible - soporte RAW limitado")
            
            def imread(self, path):
                raise NotImplementedError("rawpy no disponible - use formatos de imagen estándar")
        
        return RawpyFallback()


# Instancia global del gestor de dependencias
dependency_manager = DependencyManager()


# Funciones de conveniencia
def check_dependencies() -> Dict[str, DependencyInfo]:
    """Verifica todas las dependencias"""
    return dependency_manager.check_all_dependencies()


def safe_import(package_name: str, fallback_name: Optional[str] = None):
    """Importa un paquete de forma segura"""
    return dependency_manager.safe_import(package_name, fallback_name)


def install_missing() -> Tuple[List[str], List[str]]:
    """Instala dependencias faltantes"""
    return dependency_manager.install_missing_dependencies()


def generate_dependency_report() -> Dict[str, Any]:
    """Genera reporte de dependencias"""
    return dependency_manager.generate_report()


if __name__ == "__main__":
    # Ejecutar verificación de dependencias
    print("🔍 Verificando dependencias de SEACABAr...")
    
    report = generate_dependency_report()
    
    print(f"\n📊 Resumen:")
    print(f"  Total de dependencias: {report['total_dependencies']}")
    print(f"  Disponibles: {report['by_status']['available']}")
    print(f"  Faltantes: {report['by_status']['missing']}")
    print(f"  Errores de importación: {report['by_status']['import_error']}")
    print(f"  Problemas de versión: {report['by_status']['version_mismatch']}")
    
    # Mostrar dependencias faltantes
    missing = dependency_manager.get_missing_dependencies()
    if missing:
        print(f"\n❌ Dependencias requeridas faltantes:")
        for dep in missing:
            print(f"  - {dep}")
        
        print(f"\n🔧 Para instalar dependencias faltantes, ejecute:")
        print(f"  pip install -r requirements.txt")
    else:
        print(f"\n✅ Todas las dependencias requeridas están disponibles")