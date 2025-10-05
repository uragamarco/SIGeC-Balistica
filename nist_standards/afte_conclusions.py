"""
Sistema de Conclusiones AFTE para Análisis Balístico Forense
==========================================================

Este módulo implementa el sistema de conclusiones de la Association of Firearm 
and Tool Mark Examiners (AFTE) para análisis comparativo de evidencia balística.

Categorías AFTE:
- Identification: Suficiente acuerdo de características individuales
- Inconclusive: Algunas características en acuerdo, insuficientes para conclusión
- Elimination: Suficiente desacuerdo de características para excluir origen común

Basado en:
- AFTE Theory of Identification Range of Conclusions
- AFTE Criteria for Identification Committee Report (2011)
- SWGGUN Guidelines for Firearms and Toolmark Identification
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math
from datetime import datetime
import json


class AFTEConclusion(Enum):
    """Conclusiones AFTE estándar"""
    IDENTIFICATION = "identification"
    INCONCLUSIVE_A = "inconclusive_a"  # Algunas características en acuerdo
    INCONCLUSIVE_B = "inconclusive_b"  # Características insuficientes
    INCONCLUSIVE_C = "inconclusive_c"  # Características no reproducibles
    ELIMINATION = "elimination"
    UNSUITABLE = "unsuitable"  # Evidencia inadecuada para comparación


class ConfidenceLevel(Enum):
    """Niveles de confianza para conclusiones"""
    VERY_HIGH = "very_high"      # >95%
    HIGH = "high"                # 85-95%
    MODERATE = "moderate"        # 70-85%
    LOW = "low"                  # 50-70%
    VERY_LOW = "very_low"        # <50%


class FeatureType(Enum):
    """Tipos de características balísticas"""
    CLASS_CHARACTERISTICS = "class"          # Características de clase
    INDIVIDUAL_CHARACTERISTICS = "individual" # Características individuales
    SUBCLASS_CHARACTERISTICS = "subclass"    # Características de subclase


@dataclass
class FeatureMatch:
    """Información de coincidencia de características"""
    feature_id: str
    feature_type: FeatureType
    match_quality: float  # 0-1
    confidence: float     # 0-1
    location: Tuple[int, int]  # Coordenadas
    description: str
    examiner_notes: str = ""


@dataclass
class AFTEAnalysisResult:
    """Resultado de análisis AFTE"""
    evidence_id_1: str
    evidence_id_2: str
    conclusion: AFTEConclusion
    confidence_level: ConfidenceLevel
    confidence_score: float
    feature_matches: List[FeatureMatch]
    class_characteristics: Dict[str, Any]
    individual_characteristics: Dict[str, Any]
    subclass_characteristics: Dict[str, Any]
    examiner_id: str
    examination_date: datetime
    methodology: str
    quality_assessment: Dict[str, float]
    statistical_analysis: Dict[str, Any]
    notes: str
    review_required: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte resultado a diccionario"""
        return {
            'evidence_id_1': self.evidence_id_1,
            'evidence_id_2': self.evidence_id_2,
            'conclusion': self.conclusion.value,
            'confidence_level': self.confidence_level.value,
            'confidence_score': self.confidence_score,
            'feature_matches': [
                {
                    'feature_id': fm.feature_id,
                    'feature_type': fm.feature_type.value,
                    'match_quality': fm.match_quality,
                    'confidence': fm.confidence,
                    'location': fm.location,
                    'description': fm.description,
                    'examiner_notes': fm.examiner_notes
                }
                for fm in self.feature_matches
            ],
            'class_characteristics': self.class_characteristics,
            'individual_characteristics': self.individual_characteristics,
            'subclass_characteristics': self.subclass_characteristics,
            'examiner_id': self.examiner_id,
            'examination_date': self.examination_date.isoformat(),
            'methodology': self.methodology,
            'quality_assessment': self.quality_assessment,
            'statistical_analysis': self.statistical_analysis,
            'notes': self.notes,
            'review_required': self.review_required
        }


class AFTEConclusionEngine:
    """
    Motor de conclusiones AFTE para análisis balístico
    """
    
    def __init__(self):
        # Umbrales para conclusiones AFTE
        self.identification_thresholds = {
            'min_individual_features': 8,      # Mínimo de características individuales
            'min_match_quality': 0.85,         # Calidad mínima de coincidencia
            'min_confidence': 0.90,            # Confianza mínima
            'max_disagreement_ratio': 0.05     # Máximo ratio de desacuerdo
        }
        
        self.elimination_thresholds = {
            'max_match_quality': 0.30,         # Máxima calidad de coincidencia
            'min_disagreement_features': 3,    # Mínimo de características en desacuerdo
            'min_disagreement_ratio': 0.70     # Mínimo ratio de desacuerdo
        }
        
        self.inconclusive_thresholds = {
            'min_features_for_analysis': 3,    # Mínimo de características para análisis
            'quality_threshold': 0.50          # Umbral de calidad para inconclusivo
        }
        
        # Pesos para diferentes tipos de características
        self.feature_weights = {
            FeatureType.INDIVIDUAL_CHARACTERISTICS: 1.0,
            FeatureType.CLASS_CHARACTERISTICS: 0.3,
            FeatureType.SUBCLASS_CHARACTERISTICS: 0.6
        }
    
    def analyze_comparison(self, 
                          evidence_1_features: List[Dict[str, Any]],
                          evidence_2_features: List[Dict[str, Any]],
                          match_results: List[Dict[str, Any]],
                          evidence_id_1: str,
                          evidence_id_2: str,
                          examiner_id: str = "system",
                          methodology: str = "automated_afte") -> AFTEAnalysisResult:
        """
        Analiza comparación y determina conclusión AFTE
        
        Args:
            evidence_1_features: Características de evidencia 1
            evidence_2_features: Características de evidencia 2
            match_results: Resultados de coincidencias
            evidence_id_1: ID de evidencia 1
            evidence_id_2: ID de evidencia 2
            examiner_id: ID del examinador
            methodology: Metodología utilizada
            
        Returns:
            AFTEAnalysisResult: Resultado del análisis AFTE
        """
        try:
            # Procesar coincidencias de características
            feature_matches = self._process_feature_matches(match_results)
            
            # Clasificar características por tipo
            class_chars, individual_chars, subclass_chars = self._classify_characteristics(
                evidence_1_features, evidence_2_features, feature_matches
            )
            
            # Calcular métricas de calidad
            quality_assessment = self._assess_quality(
                evidence_1_features, evidence_2_features, feature_matches
            )
            
            # Realizar análisis estadístico
            statistical_analysis = self._perform_statistical_analysis(feature_matches)
            
            # Determinar conclusión AFTE
            conclusion, confidence_level, confidence_score = self._determine_afte_conclusion(
                feature_matches, quality_assessment, statistical_analysis
            )
            
            # Determinar si requiere revisión
            review_required = self._requires_review(
                conclusion, confidence_level, quality_assessment
            )
            
            # Generar notas automáticas
            notes = self._generate_analysis_notes(
                conclusion, feature_matches, quality_assessment
            )
            
            return AFTEAnalysisResult(
                evidence_id_1=evidence_id_1,
                evidence_id_2=evidence_id_2,
                conclusion=conclusion,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                feature_matches=feature_matches,
                class_characteristics=class_chars,
                individual_characteristics=individual_chars,
                subclass_characteristics=subclass_chars,
                examiner_id=examiner_id,
                examination_date=datetime.now(),
                methodology=methodology,
                quality_assessment=quality_assessment,
                statistical_analysis=statistical_analysis,
                notes=notes,
                review_required=review_required
            )
            
        except Exception as e:
            # Retornar resultado de error
            return self._create_error_result(
                evidence_id_1, evidence_id_2, examiner_id, str(e)
            )
    
    def _process_feature_matches(self, match_results: List[Dict[str, Any]]) -> List[FeatureMatch]:
        """
        Procesa resultados de coincidencias en objetos FeatureMatch
        
        Args:
            match_results: Lista de resultados de coincidencias
            
        Returns:
            List[FeatureMatch]: Lista de coincidencias procesadas
        """
        feature_matches = []
        
        for i, match in enumerate(match_results):
            try:
                # Determinar tipo de característica basado en la calidad y contexto
                feature_type = self._determine_feature_type(match)
                
                feature_match = FeatureMatch(
                    feature_id=match.get('feature_id', f'feature_{i}'),
                    feature_type=feature_type,
                    match_quality=float(match.get('match_quality', 0.0)),
                    confidence=float(match.get('confidence', 0.0)),
                    location=tuple(match.get('location', (0, 0))),
                    description=match.get('description', ''),
                    examiner_notes=match.get('notes', '')
                )
                
                feature_matches.append(feature_match)
                
            except Exception as e:
                print(f"Error procesando coincidencia {i}: {e}")
                continue
        
        return feature_matches
    
    def _determine_feature_type(self, match: Dict[str, Any]) -> FeatureType:
        """
        Determina el tipo de característica basado en propiedades
        
        Args:
            match: Datos de coincidencia
            
        Returns:
            FeatureType: Tipo de característica determinado
        """
        # Lógica para determinar tipo de característica
        match_quality = match.get('match_quality', 0.0)
        description = match.get('description', '').lower()
        
        # Características individuales: alta calidad, únicas
        if match_quality > 0.8 and any(keyword in description for keyword in 
                                     ['estría', 'impresión', 'marca única', 'individual']):
            return FeatureType.INDIVIDUAL_CHARACTERISTICS
        
        # Características de clase: patrones generales
        elif any(keyword in description for keyword in 
                ['calibre', 'clase', 'general', 'patrón común']):
            return FeatureType.CLASS_CHARACTERISTICS
        
        # Características de subclase: intermedias
        else:
            return FeatureType.SUBCLASS_CHARACTERISTICS
    
    def _classify_characteristics(self, 
                                evidence_1_features: List[Dict[str, Any]],
                                evidence_2_features: List[Dict[str, Any]],
                                feature_matches: List[FeatureMatch]) -> Tuple[Dict, Dict, Dict]:
        """
        Clasifica características por tipo
        
        Args:
            evidence_1_features: Características de evidencia 1
            evidence_2_features: Características de evidencia 2
            feature_matches: Coincidencias de características
            
        Returns:
            Tuple: (características_clase, características_individuales, características_subclase)
        """
        class_chars = {
            'count': len([fm for fm in feature_matches if fm.feature_type == FeatureType.CLASS_CHARACTERISTICS]),
            'avg_quality': np.mean([fm.match_quality for fm in feature_matches 
                                  if fm.feature_type == FeatureType.CLASS_CHARACTERISTICS] or [0]),
            'features': [fm for fm in feature_matches if fm.feature_type == FeatureType.CLASS_CHARACTERISTICS]
        }
        
        individual_chars = {
            'count': len([fm for fm in feature_matches if fm.feature_type == FeatureType.INDIVIDUAL_CHARACTERISTICS]),
            'avg_quality': np.mean([fm.match_quality for fm in feature_matches 
                                  if fm.feature_type == FeatureType.INDIVIDUAL_CHARACTERISTICS] or [0]),
            'features': [fm for fm in feature_matches if fm.feature_type == FeatureType.INDIVIDUAL_CHARACTERISTICS]
        }
        
        subclass_chars = {
            'count': len([fm for fm in feature_matches if fm.feature_type == FeatureType.SUBCLASS_CHARACTERISTICS]),
            'avg_quality': np.mean([fm.match_quality for fm in feature_matches 
                                  if fm.feature_type == FeatureType.SUBCLASS_CHARACTERISTICS] or [0]),
            'features': [fm for fm in feature_matches if fm.feature_type == FeatureType.SUBCLASS_CHARACTERISTICS]
        }
        
        return class_chars, individual_chars, subclass_chars
    
    def _assess_quality(self, 
                       evidence_1_features: List[Dict[str, Any]],
                       evidence_2_features: List[Dict[str, Any]],
                       feature_matches: List[FeatureMatch]) -> Dict[str, float]:
        """
        Evalúa la calidad del análisis
        
        Args:
            evidence_1_features: Características de evidencia 1
            evidence_2_features: Características de evidencia 2
            feature_matches: Coincidencias de características
            
        Returns:
            Dict: Métricas de calidad
        """
        quality_metrics = {}
        
        try:
            # Calidad promedio de coincidencias
            if feature_matches:
                quality_metrics['avg_match_quality'] = np.mean([fm.match_quality for fm in feature_matches])
                quality_metrics['max_match_quality'] = np.max([fm.match_quality for fm in feature_matches])
                quality_metrics['min_match_quality'] = np.min([fm.match_quality for fm in feature_matches])
                quality_metrics['std_match_quality'] = np.std([fm.match_quality for fm in feature_matches])
            else:
                quality_metrics['avg_match_quality'] = 0.0
                quality_metrics['max_match_quality'] = 0.0
                quality_metrics['min_match_quality'] = 0.0
                quality_metrics['std_match_quality'] = 0.0
            
            # Cobertura de características
            total_features_1 = len(evidence_1_features)
            total_features_2 = len(evidence_2_features)
            matched_features = len(feature_matches)
            
            if total_features_1 > 0 and total_features_2 > 0:
                quality_metrics['feature_coverage'] = matched_features / min(total_features_1, total_features_2)
            else:
                quality_metrics['feature_coverage'] = 0.0
            
            # Consistencia de coincidencias
            if len(feature_matches) > 1:
                qualities = [fm.match_quality for fm in feature_matches]
                quality_metrics['consistency'] = 1.0 - (np.std(qualities) / (np.mean(qualities) + 1e-10))
            else:
                quality_metrics['consistency'] = 1.0 if len(feature_matches) == 1 else 0.0
            
            # Distribución por tipos de características
            individual_count = len([fm for fm in feature_matches if fm.feature_type == FeatureType.INDIVIDUAL_CHARACTERISTICS])
            quality_metrics['individual_feature_ratio'] = individual_count / max(1, len(feature_matches))
            
            return quality_metrics
            
        except Exception as e:
            print(f"Error evaluando calidad: {e}")
            return {'avg_match_quality': 0.0, 'feature_coverage': 0.0, 'consistency': 0.0}
    
    def _perform_statistical_analysis(self, feature_matches: List[FeatureMatch]) -> Dict[str, Any]:
        """
        Realiza análisis estadístico de las coincidencias
        
        Args:
            feature_matches: Lista de coincidencias
            
        Returns:
            Dict: Resultados del análisis estadístico
        """
        stats = {}
        
        try:
            if not feature_matches:
                return {'total_matches': 0, 'statistical_significance': 0.0}
            
            # Estadísticas básicas
            qualities = [fm.match_quality for fm in feature_matches]
            confidences = [fm.confidence for fm in feature_matches]
            
            stats['total_matches'] = len(feature_matches)
            stats['mean_quality'] = np.mean(qualities)
            stats['median_quality'] = np.median(qualities)
            stats['std_quality'] = np.std(qualities)
            stats['mean_confidence'] = np.mean(confidences)
            
            # Análisis por tipo de característica
            individual_matches = [fm for fm in feature_matches if fm.feature_type == FeatureType.INDIVIDUAL_CHARACTERISTICS]
            class_matches = [fm for fm in feature_matches if fm.feature_type == FeatureType.CLASS_CHARACTERISTICS]
            subclass_matches = [fm for fm in feature_matches if fm.feature_type == FeatureType.SUBCLASS_CHARACTERISTICS]
            
            stats['individual_matches'] = len(individual_matches)
            stats['class_matches'] = len(class_matches)
            stats['subclass_matches'] = len(subclass_matches)
            
            # Significancia estadística (simplificada)
            # Basada en número y calidad de características individuales
            individual_quality = np.mean([fm.match_quality for fm in individual_matches]) if individual_matches else 0.0
            individual_weight = len(individual_matches) * individual_quality
            
            class_quality = np.mean([fm.match_quality for fm in class_matches]) if class_matches else 0.0
            class_weight = len(class_matches) * class_quality * 0.3
            
            subclass_quality = np.mean([fm.match_quality for fm in subclass_matches]) if subclass_matches else 0.0
            subclass_weight = len(subclass_matches) * subclass_quality * 0.6
            
            total_weight = individual_weight + class_weight + subclass_weight
            max_possible_weight = len(feature_matches) * 1.0  # Calidad máxima
            
            stats['statistical_significance'] = total_weight / max(1.0, max_possible_weight)
            
            # Probabilidad de coincidencia aleatoria (estimación simplificada)
            if individual_matches:
                # Probabilidad muy baja para múltiples características individuales de alta calidad
                random_match_prob = (0.1 ** len(individual_matches)) * (1.0 - individual_quality)
                stats['random_match_probability'] = min(1.0, random_match_prob)
            else:
                stats['random_match_probability'] = 0.5  # Probabilidad neutral sin características individuales
            
            return stats
            
        except Exception as e:
            print(f"Error en análisis estadístico: {e}")
            return {'total_matches': 0, 'statistical_significance': 0.0}
    
    def _determine_afte_conclusion(self, 
                                 feature_matches: List[FeatureMatch],
                                 quality_assessment: Dict[str, float],
                                 statistical_analysis: Dict[str, Any]) -> Tuple[AFTEConclusion, ConfidenceLevel, float]:
        """
        Determina la conclusión AFTE basada en el análisis
        
        Args:
            feature_matches: Lista de coincidencias
            quality_assessment: Evaluación de calidad
            statistical_analysis: Análisis estadístico
            
        Returns:
            Tuple: (conclusión, nivel_confianza, puntuación_confianza)
        """
        try:
            # Contar características por tipo
            individual_matches = [fm for fm in feature_matches if fm.feature_type == FeatureType.INDIVIDUAL_CHARACTERISTICS]
            high_quality_matches = [fm for fm in feature_matches if fm.match_quality >= self.identification_thresholds['min_match_quality']]
            
            # Métricas clave
            individual_count = len(individual_matches)
            avg_quality = quality_assessment.get('avg_match_quality', 0.0)
            max_quality = quality_assessment.get('max_match_quality', 0.0)
            statistical_significance = statistical_analysis.get('statistical_significance', 0.0)
            
            # Lógica de decisión AFTE
            
            # IDENTIFICATION: Suficientes características individuales de alta calidad
            if (individual_count >= self.identification_thresholds['min_individual_features'] and
                avg_quality >= self.identification_thresholds['min_match_quality'] and
                statistical_significance >= self.identification_thresholds['min_confidence']):
                
                confidence_score = min(0.99, (individual_count / 10.0) * avg_quality * statistical_significance)
                
                if confidence_score >= 0.95:
                    confidence_level = ConfidenceLevel.VERY_HIGH
                elif confidence_score >= 0.90:
                    confidence_level = ConfidenceLevel.HIGH
                else:
                    confidence_level = ConfidenceLevel.MODERATE
                
                return AFTEConclusion.IDENTIFICATION, confidence_level, confidence_score
            
            # ELIMINATION: Características significativamente diferentes
            elif (avg_quality <= self.elimination_thresholds['max_match_quality'] or
                  (len(feature_matches) >= self.elimination_thresholds['min_disagreement_features'] and
                   avg_quality <= 0.5)):
                
                confidence_score = min(0.99, (1.0 - avg_quality) * 0.9)
                
                if confidence_score >= 0.90:
                    confidence_level = ConfidenceLevel.VERY_HIGH
                elif confidence_score >= 0.80:
                    confidence_level = ConfidenceLevel.HIGH
                else:
                    confidence_level = ConfidenceLevel.MODERATE
                
                return AFTEConclusion.ELIMINATION, confidence_level, confidence_score
            
            # INCONCLUSIVE: Casos intermedios
            else:
                confidence_score = avg_quality * 0.7  # Confianza reducida para inconclusivos
                
                # Determinar subtipo de inconclusivo
                if individual_count > 0 and avg_quality > 0.6:
                    # Algunas características en acuerdo, pero insuficientes
                    conclusion = AFTEConclusion.INCONCLUSIVE_A
                    confidence_level = ConfidenceLevel.MODERATE
                elif len(feature_matches) < self.inconclusive_thresholds['min_features_for_analysis']:
                    # Características insuficientes para análisis
                    conclusion = AFTEConclusion.INCONCLUSIVE_B
                    confidence_level = ConfidenceLevel.LOW
                else:
                    # Características no reproducibles o de calidad variable
                    conclusion = AFTEConclusion.INCONCLUSIVE_C
                    confidence_level = ConfidenceLevel.LOW
                
                return conclusion, confidence_level, confidence_score
            
        except Exception as e:
            print(f"Error determinando conclusión AFTE: {e}")
            return AFTEConclusion.UNSUITABLE, ConfidenceLevel.VERY_LOW, 0.0
    
    def _requires_review(self, 
                        conclusion: AFTEConclusion,
                        confidence_level: ConfidenceLevel,
                        quality_assessment: Dict[str, float]) -> bool:
        """
        Determina si el resultado requiere revisión humana
        
        Args:
            conclusion: Conclusión AFTE
            confidence_level: Nivel de confianza
            quality_assessment: Evaluación de calidad
            
        Returns:
            bool: True si requiere revisión
        """
        # Siempre revisar identificaciones con confianza no muy alta
        if (conclusion == AFTEConclusion.IDENTIFICATION and 
            confidence_level not in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]):
            return True
        
        # Revisar eliminaciones con baja confianza
        if (conclusion == AFTEConclusion.ELIMINATION and 
            confidence_level == ConfidenceLevel.LOW):
            return True
        
        # Revisar casos con calidad muy variable
        if quality_assessment.get('std_match_quality', 0.0) > 0.3:
            return True
        
        # Revisar casos inconclusivos tipo A (potencial identificación)
        if conclusion == AFTEConclusion.INCONCLUSIVE_A:
            return True
        
        return False
    
    def _generate_analysis_notes(self, 
                               conclusion: AFTEConclusion,
                               feature_matches: List[FeatureMatch],
                               quality_assessment: Dict[str, float]) -> str:
        """
        Genera notas automáticas del análisis
        
        Args:
            conclusion: Conclusión AFTE
            feature_matches: Lista de coincidencias
            quality_assessment: Evaluación de calidad
            
        Returns:
            str: Notas del análisis
        """
        notes = []
        
        # Información básica
        individual_count = len([fm for fm in feature_matches if fm.feature_type == FeatureType.INDIVIDUAL_CHARACTERISTICS])
        total_matches = len(feature_matches)
        avg_quality = quality_assessment.get('avg_match_quality', 0.0)
        
        notes.append(f"Análisis automático AFTE - Total de coincidencias: {total_matches}")
        notes.append(f"Características individuales identificadas: {individual_count}")
        notes.append(f"Calidad promedio de coincidencias: {avg_quality:.3f}")
        
        # Notas específicas por conclusión
        if conclusion == AFTEConclusion.IDENTIFICATION:
            notes.append("IDENTIFICACIÓN: Suficiente acuerdo de características individuales para conclusión positiva")
            notes.append("Se observaron múltiples características individuales de alta calidad en acuerdo")
        
        elif conclusion == AFTEConclusion.ELIMINATION:
            notes.append("ELIMINACIÓN: Suficiente desacuerdo de características para excluir origen común")
            notes.append("Las características observadas son inconsistentes con origen común")
        
        elif conclusion in [AFTEConclusion.INCONCLUSIVE_A, AFTEConclusion.INCONCLUSIVE_B, AFTEConclusion.INCONCLUSIVE_C]:
            notes.append("INCONCLUSIVO: Evidencia insuficiente para determinación definitiva")
            
            if conclusion == AFTEConclusion.INCONCLUSIVE_A:
                notes.append("Algunas características en acuerdo, pero insuficientes para identificación")
            elif conclusion == AFTEConclusion.INCONCLUSIVE_B:
                notes.append("Características insuficientes para análisis comparativo")
            else:
                notes.append("Características de calidad variable o no reproducibles")
        
        # Recomendaciones
        if quality_assessment.get('feature_coverage', 0.0) < 0.5:
            notes.append("RECOMENDACIÓN: Considerar análisis adicional con mejor cobertura de características")
        
        if quality_assessment.get('consistency', 0.0) < 0.7:
            notes.append("RECOMENDACIÓN: Revisar consistencia de las coincidencias identificadas")
        
        return " | ".join(notes)
    
    def _create_error_result(self, evidence_id_1: str, evidence_id_2: str, 
                           examiner_id: str, error_msg: str) -> AFTEAnalysisResult:
        """
        Crea resultado de error
        
        Args:
            evidence_id_1: ID de evidencia 1
            evidence_id_2: ID de evidencia 2
            examiner_id: ID del examinador
            error_msg: Mensaje de error
            
        Returns:
            AFTEAnalysisResult: Resultado de error
        """
        return AFTEAnalysisResult(
            evidence_id_1=evidence_id_1,
            evidence_id_2=evidence_id_2,
            conclusion=AFTEConclusion.UNSUITABLE,
            confidence_level=ConfidenceLevel.VERY_LOW,
            confidence_score=0.0,
            feature_matches=[],
            class_characteristics={},
            individual_characteristics={},
            subclass_characteristics={},
            examiner_id=examiner_id,
            examination_date=datetime.now(),
            methodology="error",
            quality_assessment={},
            statistical_analysis={},
            notes=f"Error en análisis: {error_msg}",
            review_required=True
        )
    
    def validate_conclusion(self, result: AFTEAnalysisResult) -> Dict[str, Any]:
        """
        Valida una conclusión AFTE según criterios estándar
        
        Args:
            result: Resultado AFTE a validar
            
        Returns:
            Dict: Resultado de validación
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Validar identificación
            if result.conclusion == AFTEConclusion.IDENTIFICATION:
                individual_count = result.individual_characteristics.get('count', 0)
                avg_quality = result.quality_assessment.get('avg_match_quality', 0.0)
                
                if individual_count < self.identification_thresholds['min_individual_features']:
                    validation['warnings'].append(
                        f"Identificación con pocas características individuales ({individual_count})"
                    )
                
                if avg_quality < self.identification_thresholds['min_match_quality']:
                    validation['errors'].append(
                        f"Calidad promedio insuficiente para identificación ({avg_quality:.3f})"
                    )
                    validation['is_valid'] = False
                
                if result.confidence_level == ConfidenceLevel.LOW:
                    validation['warnings'].append("Identificación con confianza baja")
            
            # Validar eliminación
            elif result.conclusion == AFTEConclusion.ELIMINATION:
                if result.confidence_level == ConfidenceLevel.VERY_LOW:
                    validation['warnings'].append("Eliminación con confianza muy baja")
            
            # Recomendaciones generales
            if result.review_required:
                validation['recommendations'].append("Revisión humana requerida")
            
            if len(result.feature_matches) == 0:
                validation['errors'].append("No se encontraron coincidencias de características")
                validation['is_valid'] = False
            
            return validation
            
        except Exception as e:
            validation['errors'].append(f"Error en validación: {e}")
            validation['is_valid'] = False
            return validation
    
    def export_afte_report(self, result: AFTEAnalysisResult, 
                          output_path: str, format: str = 'json') -> bool:
        """
        Exporta reporte AFTE a archivo
        
        Args:
            result: Resultado AFTE
            output_path: Ruta de salida
            format: Formato ('json' o 'xml')
            
        Returns:
            bool: True si se exportó correctamente
        """
        try:
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'xml':
                # Implementar exportación XML si es necesario
                pass
            
            return True
            
        except Exception as e:
            print(f"Error exportando reporte AFTE: {e}")
            return False