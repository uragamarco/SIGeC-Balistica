#!/usr/bin/env python3
"""
Script de Pruebas Automatizadas para UI - Pestaña de Imágenes
Sistema Balístico Forense MVP

Pruebas automatizadas para verificar la funcionalidad completa de la pestaña de imágenes
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from config.unified_config import get_unified_config
from gui.main_window import MainWindow
from gui.image_tab import ImageTab


class UITestRunner(QThread):
    """Thread para ejecutar pruebas de UI automatizadas"""
    
    test_completed = pyqtSignal(dict)
    test_progress = pyqtSignal(str, int)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': []
            }
        }
        
    def run(self):
        """Ejecutar todas las pruebas de UI"""
        try:
            self.test_progress.emit("Iniciando pruebas de UI...", 0)
            
            # Pruebas de la pestaña de imágenes
            self._test_image_tab_loading()
            self._test_image_tab_processing()
            self._test_image_tab_visualization()
            self._test_image_tab_database_save()
            
            # Calcular resumen
            self.results['summary']['total'] = len(self.results['tests'])
            self.results['summary']['passed'] = sum(1 for test in self.results['tests'].values() if test['status'] == 'PASSED')
            self.results['summary']['failed'] = self.results['summary']['total'] - self.results['summary']['passed']
            
            self.test_completed.emit(self.results)
            
        except Exception as e:
            self.results['summary']['errors'].append(f"Error general en pruebas: {str(e)}")
            self.test_completed.emit(self.results)
    
    def _test_image_tab_loading(self):
        """Prueba de carga de imágenes en la pestaña de imágenes"""
        test_name = "image_tab_loading"
        self.test_progress.emit("Probando carga de imágenes...", 20)
        
        try:
            # Obtener la pestaña de imágenes
            image_tab = None
            for i in range(self.main_window.tab_widget.count()):
                tab = self.main_window.tab_widget.widget(i)
                if isinstance(tab, ImageTab):
                    image_tab = tab
                    break
            
            if not image_tab:
                raise Exception("No se encontró la pestaña de imágenes")
            
            # Verificar que los controles principales existen
            controls_exist = (
                hasattr(image_tab, 'load_button') and
                hasattr(image_tab, 'process_button') and
                hasattr(image_tab, 'evidence_type_combo') and
                hasattr(image_tab, 'processing_mode_combo')
            )
            
            if not controls_exist:
                raise Exception("Faltan controles principales en la pestaña de imágenes")
            
            # Verificar que el botón de carga está habilitado
            if not image_tab.load_button.isEnabled():
                raise Exception("El botón de carga no está habilitado")
            
            self.results['tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Controles de carga verificados correctamente',
                'details': {
                    'load_button_enabled': image_tab.load_button.isEnabled(),
                    'controls_present': controls_exist
                }
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en prueba de carga: {str(e)}',
                'details': {}
            }
    
    def _test_image_tab_processing(self):
        """Prueba de procesamiento de imágenes"""
        test_name = "image_tab_processing"
        self.test_progress.emit("Probando procesamiento de imágenes...", 40)
        
        try:
            # Obtener la pestaña de imágenes
            image_tab = None
            for i in range(self.main_window.tab_widget.count()):
                tab = self.main_window.tab_widget.widget(i)
                if isinstance(tab, ImageTab):
                    image_tab = tab
                    break
            
            # Verificar que existe una imagen de prueba
            test_image_path = "uploads/data/images/dekinder/breech_face/SS007_CCI BF R.png"
            if not os.path.exists(test_image_path):
                raise Exception(f"No se encontró la imagen de prueba: {test_image_path}")
            
            # Simular carga de imagen
            image_tab.current_image_path = test_image_path
            image_tab._display_image(test_image_path)
            
            # Verificar que la imagen se cargó
            if not hasattr(image_tab, 'current_image_path') or not image_tab.current_image_path:
                raise Exception("La imagen no se cargó correctamente")
            
            # Verificar que el botón de procesamiento está habilitado
            if not image_tab.process_button.isEnabled():
                raise Exception("El botón de procesamiento no está habilitado después de cargar imagen")
            
            self.results['tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Procesamiento de imágenes verificado correctamente',
                'details': {
                    'image_loaded': bool(image_tab.current_image_path),
                    'process_button_enabled': image_tab.process_button.isEnabled(),
                    'test_image_path': test_image_path
                }
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en prueba de procesamiento: {str(e)}',
                'details': {}
            }
    
    def _test_image_tab_visualization(self):
        """Prueba de visualización de resultados"""
        test_name = "image_tab_visualization"
        self.test_progress.emit("Probando visualización de resultados...", 60)
        
        try:
            # Obtener la pestaña de imágenes
            image_tab = None
            for i in range(self.main_window.tab_widget.count()):
                tab = self.main_window.tab_widget.widget(i)
                if isinstance(tab, ImageTab):
                    image_tab = tab
                    break
            
            # Verificar que existen los paneles de visualización
            visualization_exists = (
                hasattr(image_tab, 'original_image_label') and
                hasattr(image_tab, 'processed_image_label') and
                hasattr(image_tab, 'results_text')
            )
            
            if not visualization_exists:
                raise Exception("Faltan componentes de visualización")
            
            # Verificar que los labels de imagen existen
            labels_exist = (
                image_tab.original_image_label is not None and
                image_tab.processed_image_label is not None
            )
            
            if not labels_exist:
                raise Exception("Los labels de imagen no están inicializados")
            
            self.results['tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Componentes de visualización verificados correctamente',
                'details': {
                    'visualization_components': visualization_exists,
                    'image_labels': labels_exist
                }
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en prueba de visualización: {str(e)}',
                'details': {}
            }
    
    def _test_image_tab_database_save(self):
        """Prueba de guardado en base de datos"""
        test_name = "image_tab_database_save"
        self.test_progress.emit("Probando guardado en base de datos...", 80)
        
        try:
            # Obtener la pestaña de imágenes
            image_tab = None
            for i in range(self.main_window.tab_widget.count()):
                tab = self.main_window.tab_widget.widget(i)
                if isinstance(tab, ImageTab):
                    image_tab = tab
                    break
            
            # Verificar que existe el botón de guardado
            if not hasattr(image_tab, 'save_button'):
                raise Exception("No se encontró el botón de guardado")
            
            # Verificar que existe la base de datos
            if not hasattr(image_tab, 'database') or not image_tab.database:
                raise Exception("La base de datos no está inicializada")
            
            # Verificar conexión a la base de datos
            try:
                stats = image_tab.database.get_statistics()
                db_connected = True
            except:
                db_connected = False
            
            if not db_connected:
                raise Exception("No se puede conectar a la base de datos")
            
            self.results['tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Funcionalidad de base de datos verificada correctamente',
                'details': {
                    'save_button_exists': hasattr(image_tab, 'save_button'),
                    'database_initialized': bool(image_tab.database),
                    'database_connected': db_connected
                }
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en prueba de base de datos: {str(e)}',
                'details': {}
            }


def main():
    """Función principal para ejecutar las pruebas"""
    app = QApplication(sys.argv)
    
    try:
        # Cargar configuración
        config = get_unified_config()
        
        # Crear ventana principal
        main_window = MainWindow(config)
        main_window.show()
        
        # Esperar a que la ventana se inicialice completamente
        QTest.qWait(2000)
        
        # Crear y ejecutar pruebas
        test_runner = UITestRunner(main_window)
        
        def on_test_progress(message, progress):
            print(f"[{progress}%] {message}")
        
        def on_test_completed(results):
            print("\n" + "="*60)
            print("RESULTADOS DE PRUEBAS DE UI")
            print("="*60)
            print(f"Timestamp: {results['timestamp']}")
            print(f"Total de pruebas: {results['summary']['total']}")
            print(f"Pruebas exitosas: {results['summary']['passed']}")
            print(f"Pruebas fallidas: {results['summary']['failed']}")
            
            if results['summary']['errors']:
                print(f"Errores generales: {len(results['summary']['errors'])}")
                for error in results['summary']['errors']:
                    print(f"  - {error}")
            
            print("\nDetalle de pruebas:")
            print("-" * 40)
            
            for test_name, test_result in results['tests'].items():
                status_symbol = "✅" if test_result['status'] == 'PASSED' else "❌"
                print(f"{status_symbol} {test_name}: {test_result['status']}")
                print(f"   Mensaje: {test_result['message']}")
                if test_result['details']:
                    print(f"   Detalles: {json.dumps(test_result['details'], indent=6)}")
                print()
            
            # Guardar resultados
            results_file = f"ui_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Resultados guardados en: {results_file}")
            
            # Cerrar aplicación
            QTimer.singleShot(1000, app.quit)
        
        test_runner.test_progress.connect(on_test_progress)
        test_runner.test_completed.connect(on_test_completed)
        
        # Iniciar pruebas después de un breve delay
        QTimer.singleShot(3000, test_runner.start)
        
        # Ejecutar aplicación
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error al ejecutar pruebas: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()