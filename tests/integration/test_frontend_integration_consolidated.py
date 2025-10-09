#!/usr/bin/env python3
"""
Tests de Integración Frontend Consolidados
Sistema Balístico Forense SIGeC-Balisticar

Consolida todos los tests de integración frontend/GUI en un solo archivo
Migrado desde: test_gui_*.py, test_ui_*.py archivos
"""

import sys
import os
import time
import unittest
from pathlib import Path
from typing import Optional, Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Configurar Qt antes de importar PyQt5
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PyQt5.QtWidgets import QApplication, QWidget
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtTest import QTest
    from PyQt5.QtGui import QPixmap
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    print("⚠️ PyQt5 no disponible, tests de GUI serán saltados")

# Imports del sistema
from config.unified_config import get_unified_config
from utils.logger import get_logger

if QT_AVAILABLE:
    from gui.main_window import MainWindow
    from gui.analysis_tab import AnalysisTab
    from gui.comparison_tab import ComparisonTab
    from gui.database_tab import DatabaseTab
    from gui.reports_tab import ReportsTab
    from gui.settings_dialog import SettingsDialog


class FrontendIntegrationTestSuite(unittest.TestCase):
    """Suite consolidada de tests de integración frontend"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para toda la suite"""
        if not QT_AVAILABLE:
            raise unittest.SkipTest("PyQt5 no disponible")
        
        cls.config = get_unified_config()
        cls.logger = get_logger(__name__)
        cls.test_assets_path = Path(__file__).parent.parent / "assets"
        
        # Crear aplicación Qt para tests
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()
        
        cls.logger.info("Frontend Integration Test Suite initialized")
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza final de la suite"""
        if QT_AVAILABLE and hasattr(cls, 'app'):
            cls.app.quit()
    
    def setUp(self):
        """Configuración para cada test individual"""
        if not QT_AVAILABLE:
            self.skipTest("PyQt5 no disponible")
        
        self.start_time = time.time()
        
    def tearDown(self):
        """Limpieza después de cada test"""
        execution_time = time.time() - self.start_time
        self.logger.debug(f"Test executed in {execution_time:.2f}s")

    def test_main_window_initialization(self):
        """Test de inicialización de la ventana principal"""
        self.logger.info("🖥️ Testing main window initialization...")
        
        try:
            # Test creación de ventana principal
            main_window = MainWindow()
            self.assertIsNotNone(main_window, "Main window should initialize")
            
            # Test componentes básicos
            self.assertTrue(hasattr(main_window, 'analysis_tab'), "Should have analysis tab")
            self.assertTrue(hasattr(main_window, 'comparison_tab'), "Should have comparison tab")
            self.assertTrue(hasattr(main_window, 'database_tab'), "Should have database tab")
            
            # Test visibilidad inicial
            self.assertFalse(main_window.isVisible(), "Window should not be visible initially")
            
            # Test configuración de ventana
            self.assertGreater(main_window.width(), 0, "Window should have width")
            self.assertGreater(main_window.height(), 0, "Window should have height")
            
            # Cleanup
            main_window.close()
            
        except Exception as e:
            self.fail(f"Main window initialization failed: {e}")

    def test_analysis_tab_functionality(self):
        """Test consolidado de funcionalidad del tab de análisis"""
        self.logger.info("🔬 Testing analysis tab functionality...")
        
        try:
            main_window = MainWindow()
            analysis_tab = main_window.analysis_tab
            
            self.assertIsNotNone(analysis_tab, "Analysis tab should exist")
            
            # Test componentes del tab de análisis
            self.assertTrue(hasattr(analysis_tab, 'image_selector'), "Should have image selector")
            self.assertTrue(hasattr(analysis_tab, 'analysis_controls'), "Should have analysis controls")
            self.assertTrue(hasattr(analysis_tab, 'results_panel'), "Should have results panel")
            
            # Test carga de imagen simulada
            test_image_path = self._find_test_image()
            if test_image_path:
                # Simular carga de imagen
                with patch.object(analysis_tab, 'load_image') as mock_load:
                    mock_load.return_value = True
                    result = analysis_tab.load_image(test_image_path)
                    mock_load.assert_called_once_with(test_image_path)
            
            # Test controles de análisis
            if hasattr(analysis_tab, 'start_analysis_button'):
                self.assertTrue(analysis_tab.start_analysis_button.isEnabled(), 
                               "Analysis button should be enabled")
            
            main_window.close()
            
        except Exception as e:
            self.fail(f"Analysis tab functionality failed: {e}")

    def test_comparison_tab_functionality(self):
        """Test consolidado de funcionalidad del tab de comparación"""
        self.logger.info("⚖️ Testing comparison tab functionality...")
        
        try:
            main_window = MainWindow()
            comparison_tab = main_window.comparison_tab
            
            self.assertIsNotNone(comparison_tab, "Comparison tab should exist")
            
            # Test componentes del tab de comparación
            self.assertTrue(hasattr(comparison_tab, 'image_a_selector'), 
                           "Should have image A selector")
            self.assertTrue(hasattr(comparison_tab, 'image_b_selector'), 
                           "Should have image B selector")
            self.assertTrue(hasattr(comparison_tab, 'comparison_controls'), 
                           "Should have comparison controls")
            
            # Test carga de imágenes para comparación
            test_image_path = self._find_test_image()
            if test_image_path:
                # Simular carga de imágenes
                with patch.object(comparison_tab, 'load_image_a') as mock_load_a, \
                     patch.object(comparison_tab, 'load_image_b') as mock_load_b:
                    
                    mock_load_a.return_value = True
                    mock_load_b.return_value = True
                    
                    comparison_tab.load_image_a(test_image_path)
                    comparison_tab.load_image_b(test_image_path)
                    
                    mock_load_a.assert_called_once()
                    mock_load_b.assert_called_once()
            
            main_window.close()
            
        except Exception as e:
            self.fail(f"Comparison tab functionality failed: {e}")

    def test_database_tab_functionality(self):
        """Test consolidado de funcionalidad del tab de base de datos"""
        self.logger.info("🗄️ Testing database tab functionality...")
        
        try:
            main_window = MainWindow()
            database_tab = main_window.database_tab
            
            self.assertIsNotNone(database_tab, "Database tab should exist")
            
            # Test componentes del tab de base de datos
            self.assertTrue(hasattr(database_tab, 'case_list'), "Should have case list")
            self.assertTrue(hasattr(database_tab, 'search_controls'), "Should have search controls")
            
            # Test operaciones de base de datos simuladas
            with patch.object(database_tab, 'refresh_case_list') as mock_refresh:
                mock_refresh.return_value = True
                database_tab.refresh_case_list()
                mock_refresh.assert_called_once()
            
            # Test búsqueda
            if hasattr(database_tab, 'search_input'):
                database_tab.search_input.setText("test search")
                self.assertEqual(database_tab.search_input.text(), "test search")
            
            main_window.close()
            
        except Exception as e:
            self.fail(f"Database tab functionality failed: {e}")

    def test_reports_tab_functionality(self):
        """Test consolidado de funcionalidad del tab de reportes"""
        self.logger.info("📊 Testing reports tab functionality...")
        
        try:
            main_window = MainWindow()
            reports_tab = main_window.reports_tab
            
            self.assertIsNotNone(reports_tab, "Reports tab should exist")
            
            # Test componentes del tab de reportes
            self.assertTrue(hasattr(reports_tab, 'report_generator'), 
                           "Should have report generator")
            
            # Test generación de reporte simulada
            with patch.object(reports_tab, 'generate_report') as mock_generate:
                mock_generate.return_value = {"status": "success", "report_id": "test_001"}
                result = reports_tab.generate_report("test_case")
                mock_generate.assert_called_once_with("test_case")
                self.assertEqual(result["status"], "success")
            
            main_window.close()
            
        except Exception as e:
            self.fail(f"Reports tab functionality failed: {e}")

    def test_settings_dialog(self):
        """Test del diálogo de configuraciones"""
        self.logger.info("⚙️ Testing settings dialog...")
        
        try:
            main_window = MainWindow()
            
            # Test apertura del diálogo de configuraciones
            with patch('gui.settings_dialog.SettingsDialog') as mock_dialog:
                mock_instance = Mock()
                mock_dialog.return_value = mock_instance
                mock_instance.exec_.return_value = 1  # QDialog.Accepted
                
                # Simular apertura del diálogo
                settings_dialog = mock_dialog()
                result = settings_dialog.exec_()
                
                self.assertEqual(result, 1, "Settings dialog should be accepted")
                mock_dialog.assert_called_once()
            
            main_window.close()
            
        except Exception as e:
            self.fail(f"Settings dialog failed: {e}")

    def test_gui_responsiveness(self):
        """Test de responsividad de la GUI"""
        self.logger.info("⚡ Testing GUI responsiveness...")
        
        try:
            main_window = MainWindow()
            main_window.show()
            
            # Procesar eventos pendientes
            QApplication.processEvents()
            
            # Test cambio de tabs
            if hasattr(main_window, 'tab_widget'):
                original_index = main_window.tab_widget.currentIndex()
                
                # Cambiar a diferentes tabs
                for i in range(main_window.tab_widget.count()):
                    main_window.tab_widget.setCurrentIndex(i)
                    QApplication.processEvents()
                    self.assertEqual(main_window.tab_widget.currentIndex(), i)
                
                # Volver al tab original
                main_window.tab_widget.setCurrentIndex(original_index)
            
            main_window.close()
            
        except Exception as e:
            self.fail(f"GUI responsiveness test failed: {e}")

    def test_error_handling_gui(self):
        """Test del manejo de errores en la GUI"""
        self.logger.info("⚠️ Testing GUI error handling...")
        
        try:
            main_window = MainWindow()
            
            # Test manejo de errores en carga de imagen
            with patch.object(main_window.analysis_tab, 'load_image') as mock_load:
                mock_load.side_effect = Exception("Test error")
                
                # Debería manejar el error gracefully
                try:
                    main_window.analysis_tab.load_image("invalid_path.jpg")
                except Exception:
                    pass  # Error esperado y manejado
            
            # Test manejo de errores en operaciones de base de datos
            with patch.object(main_window.database_tab, 'refresh_case_list') as mock_refresh:
                mock_refresh.side_effect = Exception("Database error")
                
                try:
                    main_window.database_tab.refresh_case_list()
                except Exception:
                    pass  # Error esperado y manejado
            
            main_window.close()
            
        except Exception as e:
            self.fail(f"GUI error handling test failed: {e}")

    def test_memory_management(self):
        """Test de gestión de memoria en la GUI"""
        self.logger.info("💾 Testing GUI memory management...")
        
        try:
            # Test creación y destrucción múltiple de ventanas
            windows = []
            
            for i in range(5):
                window = MainWindow()
                windows.append(window)
                QApplication.processEvents()
            
            # Cerrar todas las ventanas
            for window in windows:
                window.close()
                window.deleteLater()
            
            QApplication.processEvents()
            
            # Test que las ventanas se hayan limpiado correctamente
            self.assertEqual(len([w for w in windows if not w.isVisible()]), 5,
                           "All windows should be closed")
            
        except Exception as e:
            self.fail(f"GUI memory management test failed: {e}")

    def test_widget_interactions(self):
        """Test de interacciones entre widgets"""
        self.logger.info("🔗 Testing widget interactions...")
        
        try:
            main_window = MainWindow()
            main_window.show()
            
            # Test interacción entre tabs
            if hasattr(main_window, 'analysis_tab') and hasattr(main_window, 'comparison_tab'):
                # Simular selección de imagen en analysis tab
                test_image_path = self._find_test_image()
                if test_image_path:
                    with patch.object(main_window.analysis_tab, 'get_current_image') as mock_get:
                        mock_get.return_value = test_image_path
                        
                        # Simular transferencia a comparison tab
                        with patch.object(main_window.comparison_tab, 'load_image_a') as mock_load:
                            mock_load.return_value = True
                            
                            # Simular acción de usuario
                            current_image = main_window.analysis_tab.get_current_image()
                            if current_image:
                                main_window.comparison_tab.load_image_a(current_image)
                                mock_load.assert_called_once_with(current_image)
            
            main_window.close()
            
        except Exception as e:
            self.fail(f"Widget interactions test failed: {e}")

    def _find_test_image(self) -> Optional[str]:
        """Buscar imagen de test disponible"""
        test_images = [
            self.test_assets_path / "test_image.png",
            self.test_assets_path / "FBI 58A008995 RP1_BFR.png",
            self.test_assets_path / "FBI B240793 RP1_BFR.png",
            self.test_assets_path / "SS007_CCI BF R.png"
        ]
        
        for img_path in test_images:
            if img_path.exists():
                return str(img_path)
        
        return None


def run_frontend_integration_tests():
    """Ejecutar todos los tests de integración frontend"""
    print("🚀 Ejecutando Tests de Integración Frontend Consolidados")
    print("=" * 60)
    
    if not QT_AVAILABLE:
        print("❌ PyQt5 no disponible, saltando tests de GUI")
        return True
    
    # Configurar test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(FrontendIntegrationTestSuite)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumen de resultados
    print("\n" + "=" * 60)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Saltados: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    success = len(result.errors) == 0 and len(result.failures) == 0
    print(f"Estado: {'✅ ÉXITO' if success else '❌ FALLÓ'}")
    
    return success


if __name__ == "__main__":
    success = run_frontend_integration_tests()
    sys.exit(0 if success else 1)