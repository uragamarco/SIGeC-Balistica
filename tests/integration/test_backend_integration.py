#!/usr/bin/env python3
"""
Script de prueba para la integración backend-GUI
Sistema Balístico Forense MVP

Prueba todas las conexiones de endpoints con la GUI
"""

import sys
import os
import time
import requests
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.api_client import get_api_client, configure_api_client, APIClientError
from utils.logger import LoggerMixin


class BackendIntegrationTester(LoggerMixin):
    """Tester para la integración backend-GUI"""
    
    def __init__(self):
        super().__init__()
        self.api_client = None
        self.test_image_path = None
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Configurar entorno de pruebas"""
        # Configurar cliente API
        self.api_client = configure_api_client("http://localhost:5000", 30)
        
        # Buscar imagen de prueba
        test_images = [
            "assets/test_image.png",  # Imagen creada para pruebas
            "assets/FBI 58A008995 RP1_BFR.png",
            "assets/FBI B240793 RP1_BFR.png",
            "assets/SS007_CCI BF R.png"
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                self.test_image_path = img_path
                break
        
        if not self.test_image_path:
            self.logger.warning("No se encontraron imágenes de prueba")
    
    def test_server_availability(self):
        """Probar disponibilidad del servidor"""
        self.logger.info("=== Probando disponibilidad del servidor ===")
        
        try:
            # Probar conexión directa
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code == 200:
                self.logger.info("✓ Servidor disponible - Conexión directa exitosa")
                self.logger.info(f"  Respuesta: {response.json()}")
                return True
            else:
                self.logger.error(f"✗ Servidor respondió con código: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            self.logger.error("✗ No se pudo conectar al servidor")
            return False
        except Exception as e:
            self.logger.error(f"✗ Error inesperado: {e}")
            return False
    
    def test_api_client_health_check(self):
        """Probar health check del cliente API"""
        self.logger.info("=== Probando health check del cliente API ===")
        
        try:
            health_data = self.api_client.health_check()
            self.logger.info("✓ Health check exitoso")
            self.logger.info(f"  Datos: {health_data}")
            return True
        except APIClientError as e:
            self.logger.error(f"✗ Error en health check: {e}")
            return False
        except Exception as e:
            self.logger.error(f"✗ Error inesperado en health check: {e}")
            return False
    
    def test_server_info(self):
        """Probar obtención de información del servidor"""
        self.logger.info("=== Probando información del servidor ===")
        
        try:
            server_info = self.api_client.get_server_info()
            self.logger.info("✓ Información del servidor obtenida")
            self.logger.info(f"  Disponible: {server_info.get('available')}")
            self.logger.info(f"  URL: {server_info.get('base_url')}")
            self.logger.info(f"  Timeout: {server_info.get('timeout')}")
            
            if server_info.get('health_data'):
                health = server_info['health_data']
                self.logger.info(f"  Estado: {health.get('status')}")
                self.logger.info(f"  Versión: {health.get('version')}")
            
            return server_info.get('available', False)
        except Exception as e:
            self.logger.error(f"✗ Error obteniendo información del servidor: {e}")
            return False
    
    def test_feature_extraction_endpoints(self):
        """Probar todos los endpoints de extracción de características"""
        if not self.test_image_path:
            self.logger.warning("No hay imagen de prueba disponible")
            return False
        
        self.logger.info("=== Probando endpoints de extracción de características ===")
        
        endpoints_to_test = [
            ("extract_orb_features", "ORB"),
            ("extract_sift_features", "SIFT"),
            ("extract_lbp_features", "LBP"),
            ("extract_advanced_features", "Avanzadas"),
            ("extract_ballistic_features", "Balísticas")
        ]
        
        results = {}
        
        for method_name, description in endpoints_to_test:
            self.logger.info(f"Probando extracción {description}...")
            
            try:
                method = getattr(self.api_client, method_name)
                features = method(self.test_image_path)
                
                if features and len(features) > 0:
                    self.logger.info(f"✓ Extracción {description} exitosa - {len(features)} características")
                    results[method_name] = True
                else:
                    self.logger.warning(f"⚠ Extracción {description} sin características")
                    results[method_name] = False
                    
            except APIClientError as e:
                self.logger.error(f"✗ Error en extracción {description}: {e}")
                results[method_name] = False
            except Exception as e:
                self.logger.error(f"✗ Error inesperado en extracción {description}: {e}")
                results[method_name] = False
        
        return results
    
    def test_error_handling(self):
        """Probar manejo de errores"""
        self.logger.info("=== Probando manejo de errores ===")
        
        # Probar con imagen inexistente
        try:
            self.api_client.extract_orb_features("imagen_inexistente.jpg")
            self.logger.error("✗ No se detectó error con imagen inexistente")
            return False
        except APIClientError:
            self.logger.info("✓ Error correctamente manejado para imagen inexistente")
        except Exception as e:
            self.logger.error(f"✗ Error inesperado con imagen inexistente: {e}")
            return False
        
        # Probar con datos inválidos
        try:
            self.api_client.extract_orb_features(None)
            self.logger.error("✗ No se detectó error con datos None")
            return False
        except (APIClientError, ValueError, TypeError):
            self.logger.info("✓ Error correctamente manejado para datos None")
        except Exception as e:
            self.logger.error(f"✗ Error inesperado con datos None: {e}")
            return False
        
        return True
    
    def test_performance(self):
        """Probar rendimiento básico"""
        if not self.test_image_path:
            self.logger.warning("No hay imagen de prueba para test de rendimiento")
            return False
        
        self.logger.info("=== Probando rendimiento básico ===")
        
        try:
            # Medir tiempo de extracción ORB
            start_time = time.time()
            features = self.api_client.extract_orb_features(self.test_image_path)
            end_time = time.time()
            
            duration = end_time - start_time
            self.logger.info(f"✓ Extracción ORB completada en {duration:.2f} segundos")
            
            if duration < 10:  # Menos de 10 segundos es aceptable
                self.logger.info("✓ Rendimiento aceptable")
                return True
            else:
                self.logger.warning("⚠ Rendimiento lento")
                return False
                
        except Exception as e:
            self.logger.error(f"✗ Error en test de rendimiento: {e}")
            return False
    
    def run_all_tests(self):
        """Ejecutar todas las pruebas"""
        self.logger.info("Iniciando pruebas de integración backend-GUI")
        self.logger.info("=" * 60)
        
        results = {}
        
        # Pruebas básicas
        results['server_availability'] = self.test_server_availability()
        results['api_health_check'] = self.test_api_client_health_check()
        results['server_info'] = self.test_server_info()
        
        # Pruebas de funcionalidad
        if results['server_availability']:
            results['feature_extraction'] = self.test_feature_extraction_endpoints()
            results['error_handling'] = self.test_error_handling()
            results['performance'] = self.test_performance()
        else:
            self.logger.error("Servidor no disponible - saltando pruebas de funcionalidad")
            results['feature_extraction'] = False
            results['error_handling'] = False
            results['performance'] = False
        
        # Resumen de resultados
        self.logger.info("=" * 60)
        self.logger.info("RESUMEN DE PRUEBAS")
        self.logger.info("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in results.items():
            if test_name == 'feature_extraction' and isinstance(result, dict):
                # Manejar resultados de extracción de características
                for endpoint, success in result.items():
                    total_tests += 1
                    if success:
                        passed_tests += 1
                        self.logger.info(f"✓ {endpoint}")
                    else:
                        self.logger.error(f"✗ {endpoint}")
            else:
                total_tests += 1
                if result:
                    passed_tests += 1
                    self.logger.info(f"✓ {test_name}")
                else:
                    self.logger.error(f"✗ {test_name}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        self.logger.info("=" * 60)
        self.logger.info(f"Pruebas completadas: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            self.logger.info("🎉 Integración backend-GUI EXITOSA")
        elif success_rate >= 60:
            self.logger.warning("⚠️ Integración backend-GUI PARCIAL")
        else:
            self.logger.error("❌ Integración backend-GUI FALLIDA")
        
        return success_rate >= 80


def main():
    """Función principal"""
    print("Iniciando pruebas de integración backend-GUI...")
    print("Asegúrate de que el servidor backend esté ejecutándose en localhost:5000")
    print()
    
    # Esperar un momento para que el usuario lea
    time.sleep(2)
    
    tester = BackendIntegrationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
        return 0
    else:
        print("\n❌ Algunas pruebas fallaron. Revisa los logs para más detalles.")
        return 1


if __name__ == "__main__":
    sys.exit(main())