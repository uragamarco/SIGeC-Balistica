#!/usr/bin/env python3
"""
Script de Optimización de Rendimiento del Sistema Balístico Forense MVP
Sistema Balístico Forense MVP

Este script analiza y optimiza el rendimiento del sistema basándose en:
- Análisis de tiempos de procesamiento
- Optimización de parámetros de algoritmos
- Configuración de memoria y cache
- Ajustes específicos para imágenes balísticas

Uso:
    python optimize_system_performance.py [--profile] [--config-file CONFIG]
"""

import os
import sys
import time
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import psutil
import gc

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
)

class PerformanceOptimizer:
    """Optimizador de rendimiento del sistema balístico forense"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        
        # Configuración por defecto
        self.default_config = {
            "preprocessing": {
                "resize_max_dimension": 1024,
                "gaussian_kernel_size": 5,
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": (8, 8)
            },
            "feature_extraction": {
                "orb_max_features": 500,
                "orb_scale_factor": 1.2,
                "orb_n_levels": 8,
                "orb_edge_threshold": 31,
                "orb_first_level": 0,
                "orb_wta_k": 2,
                "orb_score_type": cv2.ORB_HARRIS_SCORE,
                "orb_patch_size": 31
            },
            "matching": {
                "bf_norm_type": cv2.NORM_HAMMING,
                "bf_cross_check": True,
                "lowe_ratio": 0.75,
                "min_match_count": 10
            },
            "memory": {
                "max_image_cache": 50,
                "gc_threshold": 100,
                "memory_limit_mb": 512
            }
        }
        
        self.current_config = self.load_config()
        self.performance_metrics = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Carga configuración desde archivo o usa la por defecto"""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Configuración cargada desde: {self.config_file}")
                return {**self.default_config, **config}
            except Exception as e:
                self.logger.warning(f"Error cargando configuración: {e}. Usando configuración por defecto.")
        
        return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any], filename: str = "optimized_config.json"):
        """Guarda configuración optimizada"""
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Configuración guardada en: {filename}")
        except Exception as e:
            self.logger.error(f"Error guardando configuración: {e}")
    
    def create_test_images(self, sizes: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Crea imágenes de prueba con diferentes tamaños"""
        test_images = []
        
        for width, height in sizes:
            # Crear imagen sintética que simule características balísticas
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Agregar ruido de fondo
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            image = cv2.add(image, noise)
            
            # Agregar círculos (simulando marcas de percutor)
            center_x, center_y = width // 2, height // 2
            cv2.circle(image, (center_x, center_y), min(width, height) // 8, (200, 200, 200), -1)
            cv2.circle(image, (center_x, center_y), min(width, height) // 12, (100, 100, 100), 2)
            
            # Agregar líneas (simulando estrías)
            for i in range(5):
                angle = i * 36  # 36 grados entre líneas
                length = min(width, height) // 4
                x1 = int(center_x + length * np.cos(np.radians(angle)))
                y1 = int(center_y + length * np.sin(np.radians(angle)))
                x2 = int(center_x - length * np.cos(np.radians(angle)))
                y2 = int(center_y - length * np.sin(np.radians(angle)))
                cv2.line(image, (x1, y1), (x2, y2), (150, 150, 150), 1)
            
            # Agregar textura
            for _ in range(20):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                radius = np.random.randint(1, 5)
                color = np.random.randint(80, 180)
                cv2.circle(image, (x, y), radius, (color, color, color), -1)
            
            test_images.append(image)
        
        return test_images
    
    def benchmark_preprocessing(self, images: List[np.ndarray], config: Dict[str, Any]) -> Dict[str, float]:
        """Benchmark del preprocesamiento con diferentes configuraciones"""
        from image_processing.unified_preprocessor import UnifiedPreprocessor
        
        preprocessor = UnifiedPreprocessor()
        times = []
        memory_usage = []
        
        for image in images:
            # Medir memoria antes
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Medir tiempo de procesamiento
            start_time = time.time()
            result = preprocessor.preprocess_image(image)
            end_time = time.time()
            
            # Medir memoria después
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            if result.success:
                times.append(end_time - start_time)
                memory_usage.append(mem_after - mem_before)
            
            # Limpiar memoria
            del result
            gc.collect()
        
        return {
            "avg_time": np.mean(times) if times else 0,
            "std_time": np.std(times) if times else 0,
            "min_time": np.min(times) if times else 0,
            "max_time": np.max(times) if times else 0,
            "avg_memory": np.mean(memory_usage) if memory_usage else 0,
            "total_images": len(images),
            "success_rate": len(times) / len(images) if images else 0
        }
    
    def benchmark_feature_extraction(self, images: List[np.ndarray], config: Dict[str, Any]) -> Dict[str, float]:
        """Benchmark de extracción de características"""
        from image_processing.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        times = []
        feature_counts = []
        memory_usage = []
        
        for image in images:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            
            # Medir memoria antes
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Medir tiempo de extracción
            start_time = time.time()
            try:
                features = extractor.extract_orb_features(gray_image, 
                                                        max_features=config["feature_extraction"]["orb_max_features"])
                end_time = time.time()
                
                # Medir memoria después
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                
                times.append(end_time - start_time)
                feature_counts.append(features.get('num_keypoints', 0))
                memory_usage.append(mem_after - mem_before)
                
            except Exception as e:
                self.logger.warning(f"Error en extracción de características: {e}")
            
            # Limpiar memoria
            gc.collect()
        
        return {
            "avg_time": np.mean(times) if times else 0,
            "std_time": np.std(times) if times else 0,
            "avg_features": np.mean(feature_counts) if feature_counts else 0,
            "std_features": np.std(feature_counts) if feature_counts else 0,
            "avg_memory": np.mean(memory_usage) if memory_usage else 0,
            "success_rate": len(times) / len(images) if images else 0
        }
    
    def benchmark_matching(self, images: List[np.ndarray], config: Dict[str, Any]) -> Dict[str, float]:
        """Benchmark de matching entre imágenes"""
        from matching.unified_matcher import UnifiedMatcher
        
        matcher = UnifiedMatcher()
        times = []
        similarity_scores = []
        match_counts = []
        
        # Comparar cada imagen consigo misma y con la siguiente
        for i in range(len(images)):
            for j in range(i, min(i + 2, len(images))):
                start_time = time.time()
                try:
                    result = matcher.compare_images(images[i], images[j])
                    end_time = time.time()
                    
                    if result:
                        times.append(end_time - start_time)
                        similarity_scores.append(result.similarity_score)
                        match_counts.append(result.good_matches)
                        
                except Exception as e:
                    self.logger.warning(f"Error en matching: {e}")
                
                # Limpiar memoria
                gc.collect()
        
        return {
            "avg_time": np.mean(times) if times else 0,
            "std_time": np.std(times) if times else 0,
            "avg_similarity": np.mean(similarity_scores) if similarity_scores else 0,
            "avg_matches": np.mean(match_counts) if match_counts else 0,
            "success_rate": len(times) / (len(images) * 2) if images else 0
        }
    
    def optimize_orb_parameters(self, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """Optimiza parámetros de ORB para mejor rendimiento"""
        from image_processing.feature_extractor import FeatureExtractor
        
        self.logger.info("Optimizando parámetros de ORB...")
        
        # Parámetros a probar
        max_features_options = [250, 500, 750, 1000]
        scale_factor_options = [1.1, 1.2, 1.3]
        n_levels_options = [6, 8, 10]
        
        best_config = None
        best_score = 0
        results = []
        
        for max_features in max_features_options:
            for scale_factor in scale_factor_options:
                for n_levels in n_levels_options:
                    config = {
                        "orb_max_features": max_features,
                        "orb_scale_factor": scale_factor,
                        "orb_n_levels": n_levels
                    }
                    
                    # Probar configuración
                    extractor = FeatureExtractor()
                    times = []
                    feature_counts = []
                    
                    for image in test_images[:3]:  # Solo usar 3 imágenes para optimización
                        if len(image.shape) == 3:
                            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray_image = image
                        
                        start_time = time.time()
                        try:
                            features = extractor.extract_orb_features(gray_image, max_features=max_features)
                            end_time = time.time()
                            
                            times.append(end_time - start_time)
                            feature_counts.append(features.get('num_keypoints', 0))
                            
                        except Exception as e:
                            self.logger.warning(f"Error probando configuración ORB: {e}")
                            break
                    
                    if times and feature_counts:
                        avg_time = np.mean(times)
                        avg_features = np.mean(feature_counts)
                        
                        # Calcular score (más características en menos tiempo es mejor)
                        score = avg_features / (avg_time + 0.001)  # Evitar división por cero
                        
                        results.append({
                            "config": config,
                            "avg_time": avg_time,
                            "avg_features": avg_features,
                            "score": score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_config = config
        
        self.logger.info(f"Mejor configuración ORB encontrada: {best_config} (score: {best_score:.2f})")
        return {
            "best_config": best_config,
            "best_score": best_score,
            "all_results": results
        }
    
    def optimize_preprocessing_parameters(self, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """Optimiza parámetros de preprocesamiento"""
        from image_processing.unified_preprocessor import UnifiedPreprocessor
        
        self.logger.info("Optimizando parámetros de preprocesamiento...")
        
        # Parámetros a probar
        resize_options = [512, 768, 1024, 1280]
        clahe_clip_options = [1.0, 2.0, 3.0, 4.0]
        gaussian_kernel_options = [3, 5, 7]
        
        best_config = None
        best_score = float('inf')  # Menor tiempo es mejor
        results = []
        
        for resize_max in resize_options:
            for clahe_clip in clahe_clip_options:
                for gaussian_kernel in gaussian_kernel_options:
                    config = {
                        "resize_max_dimension": resize_max,
                        "clahe_clip_limit": clahe_clip,
                        "gaussian_kernel_size": gaussian_kernel
                    }
                    
                    # Probar configuración
                    preprocessor = UnifiedPreprocessor()
                    times = []
                    success_count = 0
                    
                    for image in test_images[:3]:  # Solo usar 3 imágenes para optimización
                        start_time = time.time()
                        try:
                            result = preprocessor.preprocess_image(image)
                            end_time = time.time()
                            
                            if result.success:
                                times.append(end_time - start_time)
                                success_count += 1
                                
                        except Exception as e:
                            self.logger.warning(f"Error probando configuración preprocesamiento: {e}")
                    
                    if times:
                        avg_time = np.mean(times)
                        success_rate = success_count / len(test_images[:3])
                        
                        # Score: tiempo promedio penalizado por fallos
                        score = avg_time / (success_rate + 0.1)
                        
                        results.append({
                            "config": config,
                            "avg_time": avg_time,
                            "success_rate": success_rate,
                            "score": score
                        })
                        
                        if score < best_score:
                            best_score = score
                            best_config = config
        
        self.logger.info(f"Mejor configuración preprocesamiento encontrada: {best_config} (score: {best_score:.3f})")
        return {
            "best_config": best_config,
            "best_score": best_score,
            "all_results": results
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Ejecuta benchmark completo del sistema"""
        self.logger.info("Iniciando benchmark completo del sistema...")
        
        # Crear imágenes de prueba
        test_sizes = [(400, 400), (600, 600), (800, 600), (1024, 768)]
        test_images = self.create_test_images(test_sizes)
        
        # Benchmark con configuración actual
        self.logger.info("Benchmarking configuración actual...")
        current_results = {
            "preprocessing": self.benchmark_preprocessing(test_images, self.current_config),
            "feature_extraction": self.benchmark_feature_extraction(test_images, self.current_config),
            "matching": self.benchmark_matching(test_images, self.current_config)
        }
        
        # Optimizar parámetros
        orb_optimization = self.optimize_orb_parameters(test_images)
        preprocessing_optimization = self.optimize_preprocessing_parameters(test_images)
        
        # Crear configuración optimizada
        optimized_config = self.current_config.copy()
        
        if orb_optimization["best_config"]:
            optimized_config["feature_extraction"].update(orb_optimization["best_config"])
        
        if preprocessing_optimization["best_config"]:
            optimized_config["preprocessing"].update(preprocessing_optimization["best_config"])
        
        # Benchmark con configuración optimizada
        self.logger.info("Benchmarking configuración optimizada...")
        optimized_results = {
            "preprocessing": self.benchmark_preprocessing(test_images, optimized_config),
            "feature_extraction": self.benchmark_feature_extraction(test_images, optimized_config),
            "matching": self.benchmark_matching(test_images, optimized_config)
        }
        
        # Calcular mejoras
        improvements = self.calculate_improvements(current_results, optimized_results)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "test_images_count": len(test_images),
            "test_sizes": test_sizes,
            "current_config": self.current_config,
            "optimized_config": optimized_config,
            "current_results": current_results,
            "optimized_results": optimized_results,
            "orb_optimization": orb_optimization,
            "preprocessing_optimization": preprocessing_optimization,
            "improvements": improvements,
            "system_info": self.get_system_info()
        }
    
    def calculate_improvements(self, current: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, float]:
        """Calcula mejoras porcentuales entre configuraciones"""
        improvements = {}
        
        for module in ["preprocessing", "feature_extraction", "matching"]:
            if module in current and module in optimized:
                current_time = current[module].get("avg_time", 0)
                optimized_time = optimized[module].get("avg_time", 0)
                
                if current_time > 0:
                    time_improvement = ((current_time - optimized_time) / current_time) * 100
                    improvements[f"{module}_time_improvement"] = time_improvement
                else:
                    improvements[f"{module}_time_improvement"] = 0
        
        return improvements
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtiene información del sistema"""
        try:
            import platform
            
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "opencv_version": cv2.__version__
            }
        except Exception as e:
            self.logger.warning(f"Error obteniendo información del sistema: {e}")
            return {}
    
    def generate_optimization_report(self, results: Dict[str, Any], output_dir: str = "optimization_results"):
        """Genera reporte de optimización"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar reporte JSON
        json_report_path = output_path / f"optimization_report_{timestamp}.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Guardar configuración optimizada
        config_path = output_path / f"optimized_config_{timestamp}.json"
        self.save_config(results["optimized_config"], str(config_path))
        
        # Generar reporte de texto
        text_report_path = output_path / f"optimization_summary_{timestamp}.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("REPORTE DE OPTIMIZACIÓN DE RENDIMIENTO\n")
            f.write("Sistema Balístico Forense MVP\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Fecha: {results['timestamp']}\n")
            f.write(f"Imágenes de prueba: {results['test_images_count']}\n")
            f.write(f"Tamaños probados: {results['test_sizes']}\n\n")
            
            # Mejoras
            f.write("MEJORAS DE RENDIMIENTO:\n")
            f.write("-" * 30 + "\n")
            for key, value in results["improvements"].items():
                f.write(f"{key}: {value:.2f}%\n")
            f.write("\n")
            
            # Configuración actual vs optimizada
            f.write("CONFIGURACIÓN ACTUAL:\n")
            f.write("-" * 30 + "\n")
            f.write(json.dumps(results["current_config"], indent=2) + "\n\n")
            
            f.write("CONFIGURACIÓN OPTIMIZADA:\n")
            f.write("-" * 30 + "\n")
            f.write(json.dumps(results["optimized_config"], indent=2) + "\n\n")
            
            # Resultados detallados
            f.write("RESULTADOS DETALLADOS:\n")
            f.write("-" * 30 + "\n")
            
            for phase in ["current_results", "optimized_results"]:
                f.write(f"\n{phase.upper()}:\n")
                for module, metrics in results[phase].items():
                    f.write(f"  {module}:\n")
                    for metric, value in metrics.items():
                        f.write(f"    {metric}: {value}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        self.logger.info(f"Reporte de optimización generado: {json_report_path}")
        return str(json_report_path)

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimizador de rendimiento del sistema balístico forense")
    parser.add_argument("--profile", action="store_true", help="Ejecutar profiling detallado")
    parser.add_argument("--config-file", help="Archivo de configuración personalizado")
    parser.add_argument("--output-dir", default="optimization_results", help="Directorio de salida")
    
    args = parser.parse_args()
    
    print("🚀 INICIANDO OPTIMIZACIÓN DE RENDIMIENTO")
    print("Sistema Balístico Forense MVP")
    print("=" * 50)
    
    # Crear optimizador
    optimizer = PerformanceOptimizer(args.config_file)
    
    try:
        # Ejecutar benchmark completo
        results = optimizer.run_comprehensive_benchmark()
        
        # Generar reporte
        report_path = optimizer.generate_optimization_report(results, args.output_dir)
        
        print("\n" + "=" * 50)
        print("📊 RESULTADOS DE OPTIMIZACIÓN")
        print("=" * 50)
        
        # Mostrar mejoras principales
        improvements = results["improvements"]
        if improvements:
            print("\n🎯 Mejoras de rendimiento:")
            for key, value in improvements.items():
                if value > 0:
                    print(f"   ✅ {key}: +{value:.1f}%")
                elif value < 0:
                    print(f"   ⚠️ {key}: {value:.1f}%")
        
        print(f"\n📄 Reporte completo: {report_path}")
        print(f"⚙️ Configuración optimizada guardada")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error durante optimización: {e}")
        logging.error(f"Error en optimización: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()