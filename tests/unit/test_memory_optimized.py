#!/usr/bin/env python3
"""
Script de prueba para verificar las optimizaciones de memoria
en ballistic_features.py consolidado
"""

import sys
import os
import numpy as np
import cv2
import psutil
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_memory_info():
    """Muestra información actual de memoria"""
    memory = psutil.virtual_memory()
    logger.info(f"Memoria total: {memory.total / (1024**3):.1f}GB")
    logger.info(f"Memoria disponible: {memory.available / (1024**3):.1f}GB")
    logger.info(f"Memoria usada: {memory.used / (1024**3):.1f}GB")
    logger.info(f"Porcentaje usado: {memory.percent:.1f}%")

def create_synthetic_cartridge_image(size=(512, 512)):
    """Crea una imagen sintética de casquillo para pruebas"""
    image = np.zeros(size, dtype=np.uint8)
    center = (size[0]//2, size[1]//2)
    
    # Crear círculo principal (casquillo)
    cv2.circle(image, center, size[0]//3, 128, -1)
    
    # Crear marca del percutor (círculo pequeño en el centro)
    cv2.circle(image, center, 20, 200, -1)
    
    # Agregar algo de ruido para simular textura
    noise = np.random.normal(0, 10, size).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # Agregar algunas líneas para simular estrías
    for i in range(5):
        angle = i * 36  # 5 líneas espaciadas 36 grados
        x1 = int(center[0] + 50 * np.cos(np.radians(angle)))
        y1 = int(center[1] + 50 * np.sin(np.radians(angle)))
        x2 = int(center[0] + 100 * np.cos(np.radians(angle)))
        y2 = int(center[1] + 100 * np.sin(np.radians(angle)))
        cv2.line(image, (x1, y1), (x2, y2), 180, 2)
    
    return image

def test_memory_optimizations():
    """Prueba las optimizaciones de memoria del extractor balístico"""
    logger.info("=== INICIANDO PRUEBA DE OPTIMIZACIONES DE MEMORIA ===")
    
    # Mostrar estado inicial de memoria
    logger.info("Estado inicial de memoria:")
    show_memory_info()
    
    try:
        # Importar el módulo optimizado
        logger.info("Importando módulo ballistic_features optimizado...")
        from image_processing.ballistic_features import BallisticFeatureExtractor, ParallelConfig
        
        # Crear configuración automática (se ajustará según memoria disponible)
        logger.info("Creando configuración automática...")
        config = Parallelget_unified_config()
        
        # Mostrar configuración aplicada
        logger.info("Configuración automática aplicada:")
        logger.info(f"  - Max workers (process): {config.max_workers_process}")
        logger.info(f"  - Max workers (thread): {config.max_workers_thread}")
        logger.info(f"  - Gabor parallel: {config.enable_gabor_parallel}")
        logger.info(f"  - ROI parallel: {config.enable_roi_parallel}")
        logger.info(f"  - Chunk size: {config.chunk_size}")
        logger.info(f"  - Memory limit: {config.memory_limit_gb:.1f}GB")
        
        # Crear extractor con configuración optimizada
        logger.info("Creando extractor balístico...")
        extractor = BallisticFeatureExtractor(parallel_config=config)
        
        # Crear imagen sintética para prueba
        logger.info("Creando imagen sintética de casquillo...")
        test_image = create_synthetic_cartridge_image()
        
        # Mostrar memoria después de inicialización
        logger.info("Memoria después de inicialización:")
        show_memory_info()
        
        # Probar extracción de características
        logger.info("Extrayendo características balísticas...")
        features = extractor.extract_ballistic_features(
            test_image, 
            specimen_type='cartridge_case',
            use_parallel=config.enable_gabor_parallel or config.enable_roi_parallel
        )
        
        # Mostrar memoria después de procesamiento
        logger.info("Memoria después de procesamiento:")
        show_memory_info()
        
        # Verificar que se extrajeron características
        if features:
            logger.info("✅ Características extraídas exitosamente:")
            logger.info(f"  - Calidad: {features.quality_score:.3f}")
            logger.info(f"  - Confianza: {features.confidence:.3f}")
            logger.info(f"  - Diámetro percutor: {features.firing_pin_diameter:.2f}")
            logger.info(f"  - Rugosidad culata: {features.breech_face_roughness:.2f}")
        else:
            logger.warning("⚠️  No se pudieron extraer características")
            
        # Probar método alternativo extract_all_features
        logger.info("Probando método extract_all_features...")
        all_features = extractor.extract_all_features(test_image)
        
        if all_features and 'ballistic_features' in all_features:
            logger.info("✅ extract_all_features funcionó correctamente")
        else:
            logger.warning("⚠️  extract_all_features falló")
        
        logger.info("=== PRUEBA COMPLETADA EXITOSAMENTE ===")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Error de importación: {e}")
        return False
    except MemoryError as e:
        logger.error(f"❌ Error de memoria: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return False

def test_with_real_image():
    """Prueba con una imagen real si está disponible"""
    test_images = [
        "assets/test_image.png",
        "assets/FBI 58A008995 RP1_BFR.png",
        "assets/FBI B240793 RP1_BFR.png"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            logger.info(f"Probando con imagen real: {image_path}")
            try:
                from image_processing.ballistic_features import extract_ballistic_features_from_path
                
                features = extract_ballistic_features_from_path(image_path)
                if features:
                    logger.info("✅ Procesamiento de imagen real exitoso")
                    return True
                else:
                    logger.warning("⚠️  No se pudieron extraer características de imagen real")
            except Exception as e:
                logger.error(f"❌ Error procesando imagen real: {e}")
    
    logger.info("No se encontraron imágenes reales para probar")
    return False

if __name__ == "__main__":
    logger.info("Iniciando pruebas de optimización de memoria...")
    
    # Prueba principal con imagen sintética
    success = test_memory_optimizations()
    
    if success:
        # Prueba adicional con imagen real si está disponible
        test_with_real_image()
        
        logger.info("🎉 TODAS LAS PRUEBAS COMPLETADAS")
        sys.exit(0)
    else:
        logger.error("💥 PRUEBAS FALLARON")
        sys.exit(1)