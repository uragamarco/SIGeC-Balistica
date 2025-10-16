#!/usr/bin/env python3
"""
Validador de Despliegue - SIGeC-Balistica
Valida que el sistema esté listo para producción
"""

import os
import sys
import subprocess
import socket
import time
import json
import logging
import psutil
import requests
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import importlib.util

# Importar configuración de producción
try:
    from production_config import get_production_config, ProductionConfig
except ImportError:
    print("Error: No se puede importar production_config")
    sys.exit(1)


@dataclass
class ValidationResult:
    """Resultado de validación"""
    category: str
    test_name: str
    status: str  # "PASS", "FAIL", "WARNING", "SKIP"
    message: str
    details: Optional[Dict[str, Any]] = None


class SystemValidator:
    """Validador de sistema"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.results: List[ValidationResult] = []
    
    def validate_system_requirements(self) -> List[ValidationResult]:
        """Validar requisitos del sistema"""
        results = []
        
        # Validar Python
        python_version = sys.version_info
        if python_version >= (3, 8):
            results.append(ValidationResult(
                "System", "Python Version", "PASS",
                f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            ))
        else:
            results.append(ValidationResult(
                "System", "Python Version", "FAIL",
                f"Python {python_version.major}.{python_version.minor} < 3.8 (required)"
            ))
        
        # Validar memoria
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 4:
            results.append(ValidationResult(
                "System", "Memory", "PASS",
                f"{memory_gb:.1f} GB available"
            ))
        else:
            results.append(ValidationResult(
                "System", "Memory", "WARNING",
                f"{memory_gb:.1f} GB available (recommended: >= 4 GB)"
            ))
        
        # Validar espacio en disco
        disk = psutil.disk_usage('/')
        disk_gb = disk.free / (1024**3)
        if disk_gb >= 10:
            results.append(ValidationResult(
                "System", "Disk Space", "PASS",
                f"{disk_gb:.1f} GB free"
            ))
        else:
            results.append(ValidationResult(
                "System", "Disk Space", "WARNING",
                f"{disk_gb:.1f} GB free (recommended: >= 10 GB)"
            ))
        
        # Validar CPU
        cpu_count = psutil.cpu_count()
        results.append(ValidationResult(
            "System", "CPU Cores", "PASS" if cpu_count >= 2 else "WARNING",
            f"{cpu_count} cores available"
        ))
        
        return results
    
    def validate_dependencies(self) -> List[ValidationResult]:
        """Validar dependencias de Python"""
        results = []
        
        required_packages = [
            "numpy", "opencv-python", "pillow", "flask", "psycopg2-binary",
            "scikit-learn", "matplotlib", "pandas", "requests", "psutil"
        ]
        
        for package in required_packages:
            try:
                spec = importlib.util.find_spec(package.replace("-", "_"))
                if spec is not None:
                    results.append(ValidationResult(
                        "Dependencies", f"Package {package}", "PASS",
                        "Installed"
                    ))
                else:
                    results.append(ValidationResult(
                        "Dependencies", f"Package {package}", "FAIL",
                        "Not installed"
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    "Dependencies", f"Package {package}", "FAIL",
                    f"Error checking: {e}"
                ))
        
        return results
    
    def validate_directories(self) -> List[ValidationResult]:
        """Validar directorios necesarios"""
        results = []
        
        directories = [
            (self.config.base_directory, "Base Directory", True),
            (self.config.data_directory, "Data Directory", True),
            (self.config.temp_directory, "Temp Directory", True),
            (self.config.log_directory, "Log Directory", True),
            (self.config.backup.backup_directory, "Backup Directory", self.config.backup.enabled)
        ]
        
        for directory, name, required in directories:
            if not required:
                results.append(ValidationResult(
                    "Directories", name, "SKIP",
                    f"Not required (feature disabled)"
                ))
                continue
            
            try:
                path = Path(directory)
                if path.exists():
                    if path.is_dir():
                        # Verificar permisos
                        if os.access(directory, os.R_OK | os.W_OK):
                            results.append(ValidationResult(
                                "Directories", name, "PASS",
                                f"Exists and writable: {directory}"
                            ))
                        else:
                            results.append(ValidationResult(
                                "Directories", name, "FAIL",
                                f"Exists but not writable: {directory}"
                            ))
                    else:
                        results.append(ValidationResult(
                            "Directories", name, "FAIL",
                            f"Path exists but is not a directory: {directory}"
                        ))
                else:
                    # Intentar crear directorio
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        results.append(ValidationResult(
                            "Directories", name, "PASS",
                            f"Created directory: {directory}"
                        ))
                    except Exception as e:
                        results.append(ValidationResult(
                            "Directories", name, "FAIL",
                            f"Cannot create directory: {e}"
                        ))
            
            except Exception as e:
                results.append(ValidationResult(
                    "Directories", name, "FAIL",
                    f"Error validating directory: {e}"
                ))
        
        return results
    
    def validate_network_connectivity(self) -> List[ValidationResult]:
        """Validar conectividad de red"""
        results = []
        
        # Validar puerto de aplicación
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.config.app_host, self.config.app_port))
            sock.close()
            
            if result == 0:
                results.append(ValidationResult(
                    "Network", "Application Port", "WARNING",
                    f"Port {self.config.app_port} is already in use"
                ))
            else:
                results.append(ValidationResult(
                    "Network", "Application Port", "PASS",
                    f"Port {self.config.app_port} is available"
                ))
        except Exception as e:
            results.append(ValidationResult(
                "Network", "Application Port", "FAIL",
                f"Error checking port: {e}"
            ))
        
        # Validar conectividad a base de datos
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((self.config.database.host, self.config.database.port))
            sock.close()
            
            if result == 0:
                results.append(ValidationResult(
                    "Network", "Database Connection", "PASS",
                    f"Can connect to {self.config.database.host}:{self.config.database.port}"
                ))
            else:
                results.append(ValidationResult(
                    "Network", "Database Connection", "FAIL",
                    f"Cannot connect to {self.config.database.host}:{self.config.database.port}"
                ))
        except Exception as e:
            results.append(ValidationResult(
                "Network", "Database Connection", "FAIL",
                f"Error checking database connection: {e}"
            ))
        
        return results


class ApplicationValidator:
    """Validador de aplicación"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
    
    def validate_configuration(self) -> List[ValidationResult]:
        """Validar configuración de aplicación"""
        results = []
        
        # Validar configuración de seguridad
        if len(self.config.security.secret_key) >= 32:
            results.append(ValidationResult(
                "Configuration", "Secret Key", "PASS",
                "Secret key is properly configured"
            ))
        else:
            results.append(ValidationResult(
                "Configuration", "Secret Key", "FAIL",
                "Secret key is too short or missing"
            ))
        
        # Validar configuración de base de datos
        db_password = os.getenv('DB_PASSWORD', self.config.database.password)
        if db_password:
            results.append(ValidationResult(
                "Configuration", "Database Password", "PASS",
                "Database password is configured"
            ))
        else:
            results.append(ValidationResult(
                "Configuration", "Database Password", "FAIL",
                "Database password is not configured"
            ))
        
        # Validar configuración de producción
        if not self.config.debug:
            results.append(ValidationResult(
                "Configuration", "Debug Mode", "PASS",
                "Debug mode is disabled"
            ))
        else:
            results.append(ValidationResult(
                "Configuration", "Debug Mode", "FAIL",
                "Debug mode should be disabled in production"
            ))
        
        # Validar HTTPS
        if self.config.security.https_only:
            results.append(ValidationResult(
                "Configuration", "HTTPS", "PASS",
                "HTTPS is enabled"
            ))
        else:
            results.append(ValidationResult(
                "Configuration", "HTTPS", "WARNING",
                "HTTPS should be enabled in production"
            ))
        
        return results
    
    def validate_core_modules(self) -> List[ValidationResult]:
        """Validar módulos principales"""
        results = []
        
        core_modules = [
            ("core.error_handler", "Error Handler"),
            ("core.fallback_system", "Fallback System"),
            ("core.intelligent_cache", "Intelligent Cache"),
            ("common.compatibility_adapter", "Compatibility Adapter"),
            ("common.nist_integration", "NIST Integration"),
            ("gui.main_window", "Main Window"),
            ("utils.logger", "Logger"),
            ("utils.fallback_implementations", "Fallback Implementations")
        ]
        
        for module_path, module_name in core_modules:
            try:
                spec = importlib.util.find_spec(module_path)
                if spec is not None:
                    results.append(ValidationResult(
                        "Modules", module_name, "PASS",
                        "Module is available"
                    ))
                else:
                    results.append(ValidationResult(
                        "Modules", module_name, "WARNING",
                        "Module not found (may use fallback)"
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    "Modules", module_name, "WARNING",
                    f"Error importing module: {e}"
                ))
        
        return results
    
    def validate_services(self) -> List[ValidationResult]:
        """Validar servicios de aplicación"""
        results = []
        
        # Validar servicio de monitoreo
        if self.config.monitoring.enabled:
            try:
                # Intentar importar el módulo de monitoreo
                spec = importlib.util.find_spec("monitoring.system_monitor")
                if spec is not None:
                    results.append(ValidationResult(
                        "Services", "Monitoring Service", "PASS",
                        "Monitoring module is available"
                    ))
                else:
                    results.append(ValidationResult(
                        "Services", "Monitoring Service", "WARNING",
                        "Monitoring module not found"
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    "Services", "Monitoring Service", "FAIL",
                    f"Error validating monitoring: {e}"
                ))
        else:
            results.append(ValidationResult(
                "Services", "Monitoring Service", "SKIP",
                "Monitoring is disabled"
            ))
        
        # Validar servicio de optimización
        try:
            spec = importlib.util.find_spec("optimization.performance_optimizer")
            if spec is not None:
                results.append(ValidationResult(
                    "Services", "Performance Optimizer", "PASS",
                    "Performance optimizer is available"
                ))
            else:
                results.append(ValidationResult(
                    "Services", "Performance Optimizer", "WARNING",
                    "Performance optimizer not found"
                ))
        except Exception as e:
            results.append(ValidationResult(
                "Services", "Performance Optimizer", "FAIL",
                f"Error validating optimizer: {e}"
            ))
        
        return results


class DeploymentValidator:
    """Validador principal de despliegue"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = get_production_config()
        self.system_validator = SystemValidator(self.config)
        self.app_validator = ApplicationValidator(self.config)
        self.all_results: List[ValidationResult] = []
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Ejecutar todas las validaciones"""
        print("🔍 Iniciando validación de despliegue...")
        
        # Ejecutar validaciones
        validations = [
            ("System Requirements", self.system_validator.validate_system_requirements),
            ("Dependencies", self.system_validator.validate_dependencies),
            ("Directories", self.system_validator.validate_directories),
            ("Network", self.system_validator.validate_network_connectivity),
            ("Configuration", self.app_validator.validate_configuration),
            ("Core Modules", self.app_validator.validate_core_modules),
            ("Services", self.app_validator.validate_services)
        ]
        
        for category, validation_func in validations:
            print(f"  Validando {category}...")
            try:
                results = validation_func()
                self.all_results.extend(results)
            except Exception as e:
                self.all_results.append(ValidationResult(
                    category, "Validation Error", "FAIL",
                    f"Error during validation: {e}"
                ))
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generar reporte de validación"""
        # Contar resultados por estado
        status_counts = {"PASS": 0, "FAIL": 0, "WARNING": 0, "SKIP": 0}
        
        for result in self.all_results:
            status_counts[result.status] += 1
        
        # Calcular puntuación
        total_tests = len(self.all_results)
        critical_tests = total_tests - status_counts["SKIP"]
        passed_tests = status_counts["PASS"]
        
        if critical_tests > 0:
            success_rate = (passed_tests / critical_tests) * 100
        else:
            success_rate = 0
        
        # Determinar estado general
        if status_counts["FAIL"] == 0:
            if status_counts["WARNING"] == 0:
                overall_status = "READY"
            else:
                overall_status = "READY_WITH_WARNINGS"
        else:
            overall_status = "NOT_READY"
        
        return {
            "overall_status": overall_status,
            "success_rate": success_rate,
            "total_tests": total_tests,
            "status_counts": status_counts,
            "results": self.all_results,
            "recommendations": self._generate_recommendations(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generar recomendaciones basadas en los resultados"""
        recommendations = []
        
        # Analizar fallos críticos
        failed_tests = [r for r in self.all_results if r.status == "FAIL"]
        
        if failed_tests:
            recommendations.append("🚨 CRÍTICO: Resolver los siguientes fallos antes del despliegue:")
            for test in failed_tests:
                recommendations.append(f"  - {test.category}: {test.test_name} - {test.message}")
        
        # Analizar advertencias
        warning_tests = [r for r in self.all_results if r.status == "WARNING"]
        
        if warning_tests:
            recommendations.append("⚠️  ADVERTENCIAS: Considerar resolver antes del despliegue:")
            for test in warning_tests:
                recommendations.append(f"  - {test.category}: {test.test_name} - {test.message}")
        
        # Recomendaciones generales
        if not failed_tests and not warning_tests:
            recommendations.append("✅ Sistema listo para despliegue en producción")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Imprimir reporte de validación"""
        print("\n" + "="*80)
        print("📋 REPORTE DE VALIDACIÓN DE DESPLIEGUE")
        print("="*80)
        
        # Estado general
        status_emoji = {
            "READY": "✅",
            "READY_WITH_WARNINGS": "⚠️",
            "NOT_READY": "❌"
        }
        
        print(f"\n{status_emoji.get(report['overall_status'], '❓')} Estado General: {report['overall_status']}")
        print(f"📊 Tasa de Éxito: {report['success_rate']:.1f}%")
        print(f"🧪 Total de Pruebas: {report['total_tests']}")
        
        # Resumen por estado
        print(f"\n📈 Resumen:")
        for status, count in report['status_counts'].items():
            emoji = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "SKIP": "⏭️"}
            print(f"  {emoji.get(status, '❓')} {status}: {count}")
        
        # Resultados detallados por categoría
        print(f"\n📋 Resultados Detallados:")
        
        categories = {}
        for result in report['results']:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        for category, results in categories.items():
            print(f"\n  📁 {category}:")
            for result in results:
                emoji = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "SKIP": "⏭️"}
                print(f"    {emoji.get(result.status, '❓')} {result.test_name}: {result.message}")
        
        # Recomendaciones
        if report['recommendations']:
            print(f"\n💡 Recomendaciones:")
            for rec in report['recommendations']:
                print(f"  {rec}")
        
        print(f"\n⏰ Validación completada: {report['timestamp']}")
        print("="*80)
    
    def save_report(self, report: Dict[str, Any], file_path: str):
        """Guardar reporte en archivo JSON"""
        # Convertir ValidationResult a dict para serialización
        serializable_results = []
        for result in report['results']:
            serializable_results.append({
                'category': result.category,
                'test_name': result.test_name,
                'status': result.status,
                'message': result.message,
                'details': result.details
            })
        
        report_copy = report.copy()
        report_copy['results'] = serializable_results
        
        with open(file_path, 'w') as f:
            json.dump(report_copy, f, indent=2)
        
        print(f"📄 Reporte guardado en: {file_path}")


def main():
    """Función principal"""
    print("🚀 SIGeC-Balistica - Validador de Despliegue")
    print("=" * 50)
    
    try:
        # Crear validador
        validator = DeploymentValidator()
        
        # Ejecutar validaciones
        report = validator.run_all_validations()
        
        # Mostrar reporte
        validator.print_report(report)
        
        # Guardar reporte
        report_file = f"/tmp/deployment_validation_{int(time.time())}.json"
        validator.save_report(report, report_file)
        
        # Código de salida basado en el estado
        if report['overall_status'] == "NOT_READY":
            sys.exit(1)
        elif report['overall_status'] == "READY_WITH_WARNINGS":
            sys.exit(2)
        else:
            sys.exit(0)
    
    except Exception as e:
        print(f"❌ Error durante la validación: {e}")
        logging.exception("Error in deployment validation")
        sys.exit(1)


if __name__ == "__main__":
    main()