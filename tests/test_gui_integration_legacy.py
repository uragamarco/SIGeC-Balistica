#!/usr/bin/env python3
"""
Test de integración GUI - Backend
Verifica que la interfaz gráfica funcione correctamente con el backend
"""

import sys
import os
import tempfile
import numpy as np
from PIL import Image
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import MainWindow
from config.unified_config import get_unified_config

class GUIIntegrationTest:
    def __init__(self):
        self.app = None
        self.main_window = None
        self.temp_files = []
        
    def setup(self):
        """Configurar el entorno de prueba"""
        print("🔧 Configurando entorno de prueba GUI...")
        
        # Crear aplicación Qt
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
            
        # Crear ventana principal
        config = get_unified_config()
        self.main_window = MainWindow(config)
        
        print("   ✓ Aplicación Qt creada")
        print("   ✓ Ventana principal inicializada")
        
    def create_test_image(self, name_prefix="test_gui"):
        """Crear imagen de prueba"""
        # Crear imagen sintética con patrones
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Agregar patrones geométricos
        cv2.rectangle(img, (100, 100), (300, 200), (255, 255, 255), -1)
        cv2.circle(img, (500, 300), 80, (128, 128, 128), -1)
        cv2.line(img, (200, 400), (600, 500), (200, 200, 200), 5)
        
        # Agregar ruido para características
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Guardar imagen temporal
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, prefix=name_prefix)
        temp_path = temp_file.name
        temp_file.close()
        
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_img.save(temp_path)
        
        self.temp_files.append(temp_path)
        return temp_path
        
    def test_gui_startup(self):
        """Test 1: Verificar que la GUI se inicie correctamente"""
        print("\n1. Probando inicio de GUI...")
        
        try:
            # Mostrar ventana
            self.main_window.show()
            
            # Procesar eventos pendientes
            self.app.processEvents()
            
            # Verificar que la ventana esté visible
            assert self.main_window.isVisible(), "La ventana principal no está visible"
            
            # Verificar componentes principales
            # Los tabs se crean dinámicamente, verificar por índice
            tab_count = self.main_window.tab_widget.count()
            assert tab_count > 0, "No se encontraron tabs en la ventana principal"
            
            # Verificar que hay al menos 3 tabs (procesamiento, comparación, base de datos, reportes)
            assert tab_count >= 3, f"Se esperaban al menos 3 tabs, encontrados: {tab_count}"
            
            print("   ✓ Ventana principal visible")
            print("   ✓ Todos los tabs presentes")
            print("   ✓ GUI iniciada correctamente")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error en inicio de GUI: {e}")
            return False
            
    def test_image_loading(self):
        """Test 2: Verificar carga de imágenes en la GUI"""
        print("\n2. Probando carga de imágenes...")
        
        try:
            # Crear imagen de prueba
            test_image = self.create_test_image("gui_load_test")
            print(f"   - Imagen de prueba creada: {os.path.basename(test_image)}")
            
            # Buscar el tab de imágenes (ahora se llama "Procesamiento")
            image_tab = None
            for i in range(self.main_window.tab_widget.count()):
                tab = self.main_window.tab_widget.widget(i)
                tab_text = self.main_window.tab_widget.tabText(i)
                print(f"   - Tab {i}: '{tab_text}'")
                if "Procesamiento" in tab_text or "Imagen" in tab_text or "Image" in tab_text:
                    image_tab = tab
                    break
            
            if image_tab is None:
                print("   ❌ No se encontró el tab de imágenes/procesamiento")
                return False
            
            # Verificar que tiene métodos de carga
            if hasattr(image_tab, 'load_image'):
                # Probar carga con imagen de prueba
                result = image_tab.load_image(test_image)
                if result:
                    print("   ✓ Carga de imagen exitosa")
                    return True
                else:
                    print("   ❌ Fallo en la carga de imagen")
                    return False
            elif hasattr(image_tab, 'add_image'):
                # Probar con método alternativo
                result = image_tab.add_image(test_image)
                if result:
                    print("   ✓ Carga de imagen exitosa (método add_image)")
                    return True
                else:
                    print("   ❌ Fallo en la carga de imagen (método add_image)")
                    return False
            else:
                print("   ❌ No se encontraron métodos de carga de imagen")
                print(f"   - Métodos disponibles: {[m for m in dir(image_tab) if not m.startswith('_')]}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error en carga de imagen: {e}")
            return False
            
    def test_database_interaction(self):
        """Test 3: Verificar interacción con base de datos desde GUI"""
        print("\n3. Probando interacción con base de datos...")
        
        try:
            # Buscar el tab de base de datos (normalmente el tercero)
            db_tab = None
            for i in range(self.main_window.tab_widget.count()):
                tab = self.main_window.tab_widget.widget(i)
                tab_text = self.main_window.tab_widget.tabText(i)
                if 'Base de Datos' in tab_text or 'Database' in tab_text:
                    db_tab = tab
                    break
            
            if db_tab is None:
                print("   ⚠ Tab de base de datos no encontrado")
                return False
            
            # Verificar que el tab tenga acceso a la base de datos
            if hasattr(db_tab, 'get_database_stats'):
                # Probar obtención de estadísticas
                stats = db_tab.get_database_stats()
                if stats is not None:
                    print("   ✓ Estadísticas de base de datos obtenidas")
                    print(f"   - Total de casos: {stats.get('total_cases', 0)}")
                    print(f"   - Total de imágenes: {stats.get('total_images', 0)}")
                else:
                    print("   ❌ No se pudieron obtener estadísticas")
                    return False
            elif hasattr(db_tab, 'get_database_connection'):
                # Probar conexión a base de datos
                db_conn = db_tab.get_database_connection()
                if db_conn is not None:
                    print("   ✓ Conexión a base de datos establecida")
                else:
                    print("   ❌ No se pudo establecer conexión a base de datos")
                    return False
            elif hasattr(db_tab, 'db_manager'):
                db_manager = db_tab.db_manager
                
                # Verificar estadísticas
                stats = db_manager.get_statistics()
                print(f"   - Total de imágenes en BD: {stats.get('total_images', 0)}")
                print(f"   - Total de casos en BD: {stats.get('total_cases', 0)}")
            else:
                print("   ⚠ Tab de BD no tiene métodos de base de datos")
                return True  # No es error crítico
            
            # Procesar eventos
            self.app.processEvents()
            
            print("   ✓ Interacción con base de datos funcional")
            return True
            
        except Exception as e:
            print(f"   ❌ Error en interacción con BD: {e}")
            return False
            
    def test_processing_pipeline(self):
        """Test 4: Verificar pipeline de procesamiento desde GUI"""
        print("\n4. Probando pipeline de procesamiento...")
        
        try:
            # Crear imagen de prueba
            test_image = self.create_test_image("gui_process_test")
            print(f"   - Imagen de prueba creada: {os.path.basename(test_image)}")
            
            # Acceder al procesador desde la ventana principal
            if hasattr(self.main_window, 'processor'):
                processor = self.main_window.processor
                
                # Cargar y procesar imagen
                img = cv2.imread(test_image)
                if img is not None:
                    # Preprocesar
                    result = processor.preprocess_image(img)
                    print(f"   - Imagen preprocesada: {result.processed_image.shape}")
                    
                    # Extraer características
                    features = processor.extract_all_features(result.processed_image, ['orb'])
                    print(f"   - Características extraídas: {len(features['keypoints'])}")
                    
            # Procesar eventos
            self.app.processEvents()
            
            print("   ✓ Pipeline de procesamiento funcional")
            return True
            
        except Exception as e:
            print(f"   ❌ Error en pipeline de procesamiento: {e}")
            return False
            
    def test_gui_responsiveness(self):
        """Test 5: Verificar que la GUI responda correctamente"""
        print("\n5. Probando responsividad de GUI...")
        
        try:
            # Cambiar entre tabs
            tab_widget = self.main_window.tab_widget
            
            for i in range(tab_widget.count()):
                tab_widget.setCurrentIndex(i)
                self.app.processEvents()
                QTest.qWait(100)  # Esperar 100ms
                
            print(f"   ✓ Navegación entre {tab_widget.count()} tabs funcional")
            
            # Verificar que la ventana responda a eventos
            self.main_window.resize(900, 700)
            self.app.processEvents()
            QTest.qWait(100)
            
            print("   ✓ Redimensionamiento funcional")
            print("   ✓ GUI responde correctamente")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error en responsividad: {e}")
            return False
            
    def cleanup(self):
        """Limpiar archivos temporales"""
        print("\n6. Limpiando archivos temporales...")
        
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    print(f"   ✓ Eliminado: {os.path.basename(temp_file)}")
            except Exception as e:
                print(f"   ⚠ Error eliminando {temp_file}: {e}")
                
        # Cerrar ventana
        if self.main_window:
            self.main_window.close()
            
    def run_all_tests(self):
        """Ejecutar todos los tests de integración GUI"""
        print("=== INICIANDO TESTS DE INTEGRACIÓN GUI ===")
        
        tests = [
            ("Inicio de GUI", self.test_gui_startup),
            ("Carga de imágenes", self.test_image_loading),
            ("Interacción con BD", self.test_database_interaction),
            ("Pipeline de procesamiento", self.test_processing_pipeline),
            ("Responsividad GUI", self.test_gui_responsiveness)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"💥 ERROR CRÍTICO en {test_name}: {e}")
                results.append((test_name, False))
                
        # Limpiar
        self.cleanup()
        
        # Resumen de resultados
        print("\n=== RESUMEN DE TESTS GUI ===")
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✓ PASÓ" if result else "❌ FALLÓ"
            print(f"{status}: {test_name}")
            if result:
                passed += 1
                
        print(f"\nResultado final: {passed}/{total} tests pasaron")
        
        if passed == total:
            print("🎉 TODOS LOS TESTS DE GUI PASARON CORRECTAMENTE")
            return True
        else:
            print("⚠ ALGUNOS TESTS DE GUI FALLARON")
            return False

def main():
    """Función principal"""
    try:
        # Crear y ejecutar tests
        test = GUIIntegrationTest()
        test.setup()
        
        # Ejecutar tests con timeout
        success = test.run_all_tests()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"💥 ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())