#!/usr/bin/env python3
"""
Test simple para aislar el problema de visualización ROI
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import matplotlib.pyplot as plt

def test_basic_matplotlib():
    """Test básico de matplotlib"""
    print("=== Test básico de matplotlib ===")
    
    try:
        # Crear imagen simple
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Crear figura
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Verificar que ax tiene imshow
        print(f"ax type: {type(ax)}")
        print(f"ax has imshow: {hasattr(ax, 'imshow')}")
        
        # Mostrar imagen
        ax.imshow(image)
        ax.set_title("Test Image")
        ax.axis('off')
        
        # Guardar
        output_path = Path("temp/test_matplotlib.png")
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Test matplotlib exitoso: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error en test matplotlib: {str(e)}")
        return False

def test_roi_visualizer_import():
    """Test de importación del ROIVisualizer"""
    print("\n=== Test importación ROIVisualizer ===")
    
    try:
        from image_processing.roi_visualizer import ROIVisualizer
        
        # Crear instancia
        visualizer = ROIVisualizer("temp/test_viz")
        print(f"✓ ROIVisualizer importado exitosamente: {type(visualizer)}")
        return True
        
    except Exception as e:
        print(f"✗ Error importando ROIVisualizer: {str(e)}")
        return False

def test_simple_roi_visualization():
    """Test simple de visualización ROI"""
    print("\n=== Test simple visualización ROI ===")
    
    try:
        from image_processing.roi_visualizer import ROIVisualizer
        
        # Crear imagen simple
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        cv2.circle(image, (100, 100), 50, (200, 200, 200), 2)
        
        # Guardar imagen temporal
        temp_image_path = Path("temp/simple_test_image.jpg")
        temp_image_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(temp_image_path), image)
        
        # ROI simple
        roi_regions = [
            {
                'bbox': [50, 50, 100, 100],
                'confidence': 0.85,
                'detection_method': 'test',
                'area': 10000,
                'center': [100, 100]
            }
        ]
        
        # Crear visualizador
        visualizer = ROIVisualizer("temp/simple_viz")
        
        # Generar solo overview
        print("Generando visualización overview...")
        overview_path = visualizer._generate_roi_overview(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            roi_regions,
            "test",
            "simple_test"
        )
        
        print(f"✓ Visualización overview generada: {overview_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error en visualización simple: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("=== Test Simple de Visualización ROI ===\n")
    
    results = []
    
    # Test 1: Matplotlib básico
    results.append(("Matplotlib básico", test_basic_matplotlib()))
    
    # Test 2: Importación ROIVisualizer
    results.append(("Importación ROIVisualizer", test_roi_visualizer_import()))
    
    # Test 3: Visualización simple
    results.append(("Visualización simple", test_simple_roi_visualization()))
    
    # Resumen
    print("\n=== RESUMEN ===")
    passed = 0
    for test_name, result in results:
        status = "✓ PASADO" if result else "✗ FALLIDO"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests pasados: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 Todos los tests pasaron!")
        return 0
    else:
        print("❌ Algunos tests fallaron")
        return 1

if __name__ == "__main__":
    sys.exit(main())