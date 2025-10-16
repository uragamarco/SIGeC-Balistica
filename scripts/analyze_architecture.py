#!/usr/bin/env python3
"""
Analizador de Arquitectura del Sistema SIGeC-Balistica
====================================================

Este script analiza la arquitectura del sistema para identificar:
- Dependencias circulares
- Patrones de importación problemáticos
- Oportunidades de optimización
- Métricas de complejidad
- Recomendaciones de mejora

Autor: SIGeC-Balistica Team
Fecha: 2024
"""

import os
import ast
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModuleInfo:
    """Información de un módulo"""
    name: str
    path: Path
    imports: List[str] = field(default_factory=list)
    from_imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    lines_of_code: int = 0
    complexity_score: float = 0.0

@dataclass
class ArchitectureAnalysis:
    """Resultado del análisis de arquitectura"""
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class ArchitectureAnalyzer:
    """Analizador de arquitectura del sistema"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.modules: Dict[str, ModuleInfo] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze(self) -> ArchitectureAnalysis:
        """Ejecuta el análisis completo de arquitectura"""
        logger.info("Iniciando análisis de arquitectura...")
        
        # 1. Escanear todos los módulos Python
        self._scan_modules()
        
        # 2. Construir grafo de dependencias
        self._build_dependency_graph()
        
        # 3. Detectar dependencias circulares
        circular_deps = self._detect_circular_dependencies()
        
        # 4. Calcular métricas de complejidad
        complexity_metrics = self._calculate_complexity_metrics()
        
        # 5. Generar recomendaciones
        recommendations = self._generate_recommendations(circular_deps, complexity_metrics)
        
        return ArchitectureAnalysis(
            modules=self.modules,
            circular_dependencies=circular_deps,
            dependency_graph=dict(self.dependency_graph),
            complexity_metrics=complexity_metrics,
            recommendations=recommendations
        )
    
    def _scan_modules(self):
        """Escanea todos los módulos Python del proyecto"""
        logger.info("Escaneando módulos Python...")
        
        for py_file in self.project_root.rglob("*.py"):
            # Filtrar archivos no deseados
            if any(exclude in str(py_file) for exclude in [
                "__pycache__", ".git", "venv", ".venv", "env", ".env",
                "site-packages", "dist", "build", ".pytest_cache"
            ]):
                continue
                
            try:
                module_info = self._analyze_module(py_file)
                if module_info:
                    self.modules[module_info.name] = module_info
            except Exception as e:
                logger.warning(f"Error analizando {py_file}: {e}")
    
    def _analyze_module(self, file_path: Path) -> Optional[ModuleInfo]:
        """Analiza un módulo individual"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parsear AST
            tree = ast.parse(content)
            
            # Obtener nombre del módulo relativo al proyecto
            rel_path = file_path.relative_to(self.project_root)
            module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')
            
            module_info = ModuleInfo(
                name=module_name,
                path=file_path,
                lines_of_code=len(content.splitlines())
            )
            
            # Analizar imports y estructura
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_info.imports.append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_info.from_imports.append(node.module)
                
                elif isinstance(node, ast.ClassDef):
                    module_info.classes.append(node.name)
                
                elif isinstance(node, ast.FunctionDef):
                    module_info.functions.append(node.name)
            
            # Calcular score de complejidad básico
            module_info.complexity_score = self._calculate_module_complexity(tree)
            
            return module_info
            
        except Exception as e:
            logger.error(f"Error procesando {file_path}: {e}")
            return None
    
    def _calculate_module_complexity(self, tree: ast.AST) -> float:
        """Calcula un score de complejidad para el módulo"""
        complexity = 0
        
        for node in ast.walk(tree):
            # Complejidad ciclomática básica
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 1
            elif isinstance(node, ast.ClassDef):
                complexity += 2
        
        return complexity
    
    def _build_dependency_graph(self):
        """Construye el grafo de dependencias entre módulos"""
        logger.info("Construyendo grafo de dependencias...")
        
        for module_name, module_info in self.modules.items():
            # Procesar imports directos
            for import_name in module_info.imports:
                if self._is_internal_module(import_name):
                    self.dependency_graph[module_name].add(import_name)
            
            # Procesar from imports
            for from_import in module_info.from_imports:
                if self._is_internal_module(from_import):
                    self.dependency_graph[module_name].add(from_import)
    
    def _is_internal_module(self, module_name: str) -> bool:
        """Verifica si un módulo es interno al proyecto"""
        # Lista de módulos internos conocidos
        internal_prefixes = [
            'core', 'utils', 'gui', 'image_processing', 'matching',
            'database', 'nist_standards', 'common', 'config',
            'performance', 'deep_learning', 'tests'
        ]
        
        return any(module_name.startswith(prefix) for prefix in internal_prefixes)
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detecta dependencias circulares usando DFS"""
        logger.info("Detectando dependencias circulares...")
        
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Encontramos un ciclo - buscar donde empieza
                try:
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:] + [node]
                    cycles.append(cycle)
                except ValueError:
                    # El nodo no está en el path actual, crear ciclo simple
                    cycles.append([node, node])
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor in self.modules:  # Solo procesar módulos conocidos
                    if dfs(neighbor, path.copy()):  # Usar copia del path
                        pass  # Continuar buscando más ciclos
            
            rec_stack.remove(node)
            if path and path[-1] == node:
                path.pop()
            return False
        
        for module in self.modules:
            if module not in visited:
                dfs(module, [])
        
        return cycles
    
    def _calculate_complexity_metrics(self) -> Dict[str, float]:
        """Calcula métricas de complejidad del sistema"""
        logger.info("Calculando métricas de complejidad...")
        
        metrics = {}
        
        # Complejidad total
        total_complexity = sum(m.complexity_score for m in self.modules.values())
        metrics['total_complexity'] = total_complexity
        
        # Complejidad promedio por módulo
        if self.modules:
            metrics['avg_complexity_per_module'] = total_complexity / len(self.modules)
        
        # Líneas de código totales
        total_loc = sum(m.lines_of_code for m in self.modules.values())
        metrics['total_lines_of_code'] = total_loc
        
        # Densidad de dependencias
        total_deps = sum(len(deps) for deps in self.dependency_graph.values())
        if self.modules:
            metrics['dependency_density'] = total_deps / len(self.modules)
        
        # Módulos más complejos
        if self.modules:
            max_complexity = max(m.complexity_score for m in self.modules.values())
            metrics['max_module_complexity'] = max_complexity
        
        return metrics
    
    def _generate_recommendations(self, circular_deps: List[List[str]], 
                                 complexity_metrics: Dict[str, float]) -> List[str]:
        """Genera recomendaciones de mejora"""
        recommendations = []
        
        # Recomendaciones basadas en dependencias circulares
        if len(circular_deps) > 0:
            recommendations.append(
                f"🔄 Se detectaron {len(circular_deps)} dependencias circulares. "
                "Considere refactorizar para eliminarlas."
            )
        
        # Recomendaciones basadas en complejidad
        complex_modules = [
            (name, info) for name, info in self.modules.items()
            if info.complexity_score > 50
        ]
        
        if complex_modules:
            recommendations.append(
                f"⚠️ {len(complex_modules)} módulos tienen alta complejidad (>50). "
                "Considere dividirlos en módulos más pequeños."
            )
        
        # Recomendaciones basadas en líneas de código
        large_modules = [
            (name, info) for name, info in self.modules.items()
            if info.lines_of_code > 1000
        ]
        
        if large_modules:
            recommendations.append(
                f"📏 {len(large_modules)} módulos tienen más de 1000 líneas. "
                "Considere dividirlos para mejorar mantenibilidad."
            )
        
        # Recomendaciones sobre patrones de diseño
        recommendations.append(
            "🏗️ Considere implementar patrones como Factory, Strategy o Observer "
            "para mejorar la flexibilidad del sistema."
        )
        
        recommendations.append(
            "🔧 Implemente interfaces abstractas para mejorar la testabilidad "
            "y reducir el acoplamiento entre módulos."
        )
        
        return recommendations

def main():
    """Función principal"""
    project_root = Path(__file__).parent.parent
    
    print("🔍 Analizando arquitectura del sistema SIGeC-Balistica...")
    
    analyzer = ArchitectureAnalyzer(project_root)
    analysis = analyzer.analyze()
    
    # Mostrar resultados
    print(f"\n📊 RESUMEN DEL ANÁLISIS")
    print("=" * 50)
    print(f"Total de módulos: {len(analysis.modules)}")
    print(f"Dependencias circulares: {len(analysis.circular_dependencies)}")
    print(f"Complejidad total: {analysis.complexity_metrics.get('total_complexity', 0):.1f}")
    print(f"Líneas de código: {analysis.complexity_metrics.get('total_lines_of_code', 0)}")
    
    # Mostrar dependencias circulares
    if analysis.circular_dependencies:
        print(f"\n🔄 DEPENDENCIAS CIRCULARES DETECTADAS:")
        for i, cycle in enumerate(analysis.circular_dependencies, 1):
            print(f"  {i}. {' -> '.join(cycle)}")
    
    # Mostrar módulos más complejos
    complex_modules = sorted(
        analysis.modules.items(),
        key=lambda x: x[1].complexity_score,
        reverse=True
    )[:10]
    
    print(f"\n⚠️ MÓDULOS MÁS COMPLEJOS:")
    for name, info in complex_modules:
        print(f"  {name}: {info.complexity_score:.1f} (LOC: {info.lines_of_code})")
    
    # Mostrar recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    for rec in analysis.recommendations:
        print(f"  {rec}")
    
    # Guardar reporte detallado
    report_path = project_root / "architecture_analysis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        # Convertir a formato serializable
        serializable_analysis = {
            'modules': {
                name: {
                    'name': info.name,
                    'path': str(info.path),
                    'imports': info.imports,
                    'from_imports': info.from_imports,
                    'classes': info.classes,
                    'functions': info.functions,
                    'lines_of_code': info.lines_of_code,
                    'complexity_score': info.complexity_score
                }
                for name, info in analysis.modules.items()
            },
            'circular_dependencies': analysis.circular_dependencies,
            'dependency_graph': {
                k: list(v) for k, v in analysis.dependency_graph.items()
            },
            'complexity_metrics': analysis.complexity_metrics,
            'recommendations': analysis.recommendations
        }
        json.dump(serializable_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Reporte detallado guardado en: {report_path}")

if __name__ == "__main__":
    main()