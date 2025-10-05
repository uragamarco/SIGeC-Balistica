"""
Implementación del Schema XML NIST para Datos Balísticos
========================================================

Este módulo implementa el formato de datos XML según los estándares NIST
para evidencia balística, incluyendo metadatos, características y resultados
de análisis.

Basado en:
- NIST Special Publication 800-101 Rev. 1
- AFTE Theory and Practice of Firearm Identification
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum


class EvidenceType(Enum):
    """Tipos de evidencia balística según NIST"""
    CARTRIDGE_CASE = "cartridge_case"
    BULLET = "bullet"
    PROJECTILE = "projectile"
    UNKNOWN = "unknown"


class ExaminationMethod(Enum):
    """Métodos de examinación según NIST"""
    MICROSCOPY = "microscopy"
    DIGITAL_IMAGING = "digital_imaging"
    AUTOMATED_COMPARISON = "automated_comparison"
    MANUAL_COMPARISON = "manual_comparison"


@dataclass
class NISTMetadata:
    """Metadatos según estándares NIST"""
    case_id: str
    evidence_id: str
    examiner_id: str
    laboratory_id: str
    examination_date: str
    submission_date: str
    evidence_type: EvidenceType
    examination_method: ExaminationMethod
    chain_of_custody: List[Dict[str, str]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        data = asdict(self)
        data['evidence_type'] = self.evidence_type.value
        data['examination_method'] = self.examination_method.value
        return data


@dataclass
class NISTImageData:
    """Datos de imagen según NIST"""
    image_id: str
    file_path: str
    acquisition_date: str
    resolution_x: int
    resolution_y: int
    bit_depth: int
    magnification: float
    lighting_conditions: str
    camera_settings: Dict[str, Any]
    calibration_data: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return asdict(self)


@dataclass
class NISTFeatureData:
    """Datos de características según NIST"""
    feature_id: str
    feature_type: str
    coordinates: List[Dict[str, float]]
    descriptors: List[float]
    confidence_score: float
    extraction_method: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return asdict(self)


@dataclass
class NISTComparisonResult:
    """Resultado de comparación según NIST"""
    comparison_id: str
    reference_evidence_id: str
    questioned_evidence_id: str
    similarity_score: float
    match_points: List[Dict[str, Any]]
    conclusion: str
    confidence_level: str
    examiner_notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return asdict(self)


class NISTSchema:
    """
    Implementación del Schema XML NIST para datos balísticos
    """
    
    def __init__(self):
        self.schema_version = "1.0"
        self.namespace = "http://nist.gov/ballistics/schema/v1.0"
        
    def create_root_element(self) -> ET.Element:
        """Crea el elemento raíz del XML NIST"""
        root = ET.Element("BallisticEvidence")
        root.set("xmlns", self.namespace)
        root.set("version", self.schema_version)
        root.set("created", datetime.now().isoformat())
        return root
    
    def add_metadata(self, root: ET.Element, metadata: NISTMetadata) -> ET.Element:
        """Añade metadatos al XML"""
        metadata_elem = ET.SubElement(root, "Metadata")
        
        # Información del caso
        case_info = ET.SubElement(metadata_elem, "CaseInformation")
        ET.SubElement(case_info, "CaseID").text = metadata.case_id
        ET.SubElement(case_info, "EvidenceID").text = metadata.evidence_id
        ET.SubElement(case_info, "SubmissionDate").text = metadata.submission_date
        
        # Información del examinador
        examiner_info = ET.SubElement(metadata_elem, "ExaminerInformation")
        ET.SubElement(examiner_info, "ExaminerID").text = metadata.examiner_id
        ET.SubElement(examiner_info, "LaboratoryID").text = metadata.laboratory_id
        ET.SubElement(examiner_info, "ExaminationDate").text = metadata.examination_date
        
        # Información de la evidencia
        evidence_info = ET.SubElement(metadata_elem, "EvidenceInformation")
        ET.SubElement(evidence_info, "EvidenceType").text = metadata.evidence_type.value
        ET.SubElement(evidence_info, "ExaminationMethod").text = metadata.examination_method.value
        
        # Cadena de custodia
        custody_elem = ET.SubElement(metadata_elem, "ChainOfCustody")
        for entry in metadata.chain_of_custody:
            custody_entry = ET.SubElement(custody_elem, "CustodyEntry")
            for key, value in entry.items():
                ET.SubElement(custody_entry, key).text = str(value)
        
        return metadata_elem
    
    def add_image_data(self, root: ET.Element, image_data: NISTImageData) -> ET.Element:
        """Añade datos de imagen al XML"""
        image_elem = ET.SubElement(root, "ImageData")
        
        # Información básica de la imagen
        ET.SubElement(image_elem, "ImageID").text = image_data.image_id
        ET.SubElement(image_elem, "FilePath").text = image_data.file_path
        ET.SubElement(image_elem, "AcquisitionDate").text = image_data.acquisition_date
        
        # Propiedades técnicas
        technical_props = ET.SubElement(image_elem, "TechnicalProperties")
        ET.SubElement(technical_props, "ResolutionX").text = str(image_data.resolution_x)
        ET.SubElement(technical_props, "ResolutionY").text = str(image_data.resolution_y)
        ET.SubElement(technical_props, "BitDepth").text = str(image_data.bit_depth)
        ET.SubElement(technical_props, "Magnification").text = str(image_data.magnification)
        ET.SubElement(technical_props, "LightingConditions").text = image_data.lighting_conditions
        
        # Configuración de cámara
        camera_elem = ET.SubElement(image_elem, "CameraSettings")
        for key, value in image_data.camera_settings.items():
            ET.SubElement(camera_elem, key).text = str(value)
        
        # Datos de calibración
        calibration_elem = ET.SubElement(image_elem, "CalibrationData")
        for key, value in image_data.calibration_data.items():
            ET.SubElement(calibration_elem, key).text = str(value)
        
        return image_elem
    
    def add_feature_data(self, root: ET.Element, features: List[NISTFeatureData]) -> ET.Element:
        """Añade datos de características al XML"""
        features_elem = ET.SubElement(root, "FeatureData")
        
        for feature in features:
            feature_elem = ET.SubElement(features_elem, "Feature")
            ET.SubElement(feature_elem, "FeatureID").text = feature.feature_id
            ET.SubElement(feature_elem, "FeatureType").text = feature.feature_type
            ET.SubElement(feature_elem, "ExtractionMethod").text = feature.extraction_method
            ET.SubElement(feature_elem, "ConfidenceScore").text = str(feature.confidence_score)
            
            # Coordenadas
            coords_elem = ET.SubElement(feature_elem, "Coordinates")
            for coord in feature.coordinates:
                coord_elem = ET.SubElement(coords_elem, "Point")
                for key, value in coord.items():
                    ET.SubElement(coord_elem, key).text = str(value)
            
            # Descriptores
            descriptors_elem = ET.SubElement(feature_elem, "Descriptors")
            for i, desc in enumerate(feature.descriptors):
                desc_elem = ET.SubElement(descriptors_elem, "Descriptor")
                desc_elem.set("index", str(i))
                desc_elem.text = str(desc)
            
            # Parámetros
            params_elem = ET.SubElement(feature_elem, "Parameters")
            for key, value in feature.parameters.items():
                ET.SubElement(params_elem, key).text = str(value)
        
        return features_elem
    
    def add_comparison_results(self, root: ET.Element, results: List[NISTComparisonResult]) -> ET.Element:
        """Añade resultados de comparación al XML"""
        results_elem = ET.SubElement(root, "ComparisonResults")
        
        for result in results:
            result_elem = ET.SubElement(results_elem, "ComparisonResult")
            ET.SubElement(result_elem, "ComparisonID").text = result.comparison_id
            ET.SubElement(result_elem, "ReferenceEvidenceID").text = result.reference_evidence_id
            ET.SubElement(result_elem, "QuestionedEvidenceID").text = result.questioned_evidence_id
            ET.SubElement(result_elem, "SimilarityScore").text = str(result.similarity_score)
            ET.SubElement(result_elem, "Conclusion").text = result.conclusion
            ET.SubElement(result_elem, "ConfidenceLevel").text = result.confidence_level
            ET.SubElement(result_elem, "ExaminerNotes").text = result.examiner_notes
            
            # Puntos de coincidencia
            matches_elem = ET.SubElement(result_elem, "MatchPoints")
            for match in result.match_points:
                match_elem = ET.SubElement(matches_elem, "MatchPoint")
                for key, value in match.items():
                    ET.SubElement(match_elem, key).text = str(value)
        
        return results_elem


class NISTDataExporter:
    """
    Exportador de datos en formato XML NIST
    """
    
    def __init__(self):
        self.schema = NISTSchema()
    
    def export_to_xml(self, 
                     metadata: NISTMetadata,
                     image_data: Optional[NISTImageData] = None,
                     features: Optional[List[NISTFeatureData]] = None,
                     comparison_results: Optional[List[NISTComparisonResult]] = None,
                     output_path: str = None) -> str:
        """
        Exporta datos a formato XML NIST
        
        Args:
            metadata: Metadatos de la evidencia
            image_data: Datos de imagen (opcional)
            features: Lista de características (opcional)
            comparison_results: Resultados de comparación (opcional)
            output_path: Ruta de salida (opcional)
            
        Returns:
            str: XML formateado como string
        """
        # Crear elemento raíz
        root = self.schema.create_root_element()
        
        # Añadir metadatos
        self.schema.add_metadata(root, metadata)
        
        # Añadir datos de imagen si están disponibles
        if image_data:
            self.schema.add_image_data(root, image_data)
        
        # Añadir características si están disponibles
        if features:
            self.schema.add_feature_data(root, features)
        
        # Añadir resultados de comparación si están disponibles
        if comparison_results:
            self.schema.add_comparison_results(root, comparison_results)
        
        # Formatear XML
        xml_str = self._prettify_xml(root)
        
        # Guardar archivo si se especifica ruta
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
        
        return xml_str
    
    def _prettify_xml(self, elem: ET.Element) -> str:
        """Formatea el XML para mejor legibilidad"""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def export_to_json(self, 
                      metadata: NISTMetadata,
                      image_data: Optional[NISTImageData] = None,
                      features: Optional[List[NISTFeatureData]] = None,
                      comparison_results: Optional[List[NISTComparisonResult]] = None,
                      output_path: str = None) -> Dict[str, Any]:
        """
        Exporta datos a formato JSON (alternativo)
        
        Args:
            metadata: Metadatos de la evidencia
            image_data: Datos de imagen (opcional)
            features: Lista de características (opcional)
            comparison_results: Resultados de comparación (opcional)
            output_path: Ruta de salida (opcional)
            
        Returns:
            Dict: Datos en formato JSON
        """
        data = {
            "schema_version": self.schema.schema_version,
            "namespace": self.schema.namespace,
            "created": datetime.now().isoformat(),
            "metadata": metadata.to_dict()
        }
        
        if image_data:
            data["image_data"] = image_data.to_dict()
        
        if features:
            data["features"] = [f.to_dict() for f in features]
        
        if comparison_results:
            data["comparison_results"] = [r.to_dict() for r in comparison_results]
        
        # Guardar archivo si se especifica ruta
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return data


class NISTDataImporter:
    """
    Importador de datos desde formato XML NIST
    """
    
    def __init__(self):
        self.schema = NISTSchema()
    
    def import_from_xml(self, xml_path: str) -> Dict[str, Any]:
        """
        Importa datos desde archivo XML NIST
        
        Args:
            xml_path: Ruta del archivo XML
            
        Returns:
            Dict: Datos importados
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        data = {
            "schema_version": root.get("version"),
            "namespace": root.get("xmlns"),
            "created": root.get("created")
        }
        
        # Importar metadatos
        metadata_elem = root.find("Metadata")
        if metadata_elem is not None:
            data["metadata"] = self._parse_metadata(metadata_elem)
        
        # Importar datos de imagen
        image_elem = root.find("ImageData")
        if image_elem is not None:
            data["image_data"] = self._parse_image_data(image_elem)
        
        # Importar características
        features_elem = root.find("FeatureData")
        if features_elem is not None:
            data["features"] = self._parse_features(features_elem)
        
        # Importar resultados de comparación
        results_elem = root.find("ComparisonResults")
        if results_elem is not None:
            data["comparison_results"] = self._parse_comparison_results(results_elem)
        
        return data
    
    def _parse_metadata(self, metadata_elem: ET.Element) -> Dict[str, Any]:
        """Parsea metadatos desde XML"""
        metadata = {}
        
        # Información del caso
        case_info = metadata_elem.find("CaseInformation")
        if case_info is not None:
            metadata["case_id"] = self._get_text(case_info, "CaseID")
            metadata["evidence_id"] = self._get_text(case_info, "EvidenceID")
            metadata["submission_date"] = self._get_text(case_info, "SubmissionDate")
        
        # Información del examinador
        examiner_info = metadata_elem.find("ExaminerInformation")
        if examiner_info is not None:
            metadata["examiner_id"] = self._get_text(examiner_info, "ExaminerID")
            metadata["laboratory_id"] = self._get_text(examiner_info, "LaboratoryID")
            metadata["examination_date"] = self._get_text(examiner_info, "ExaminationDate")
        
        # Información de la evidencia
        evidence_info = metadata_elem.find("EvidenceInformation")
        if evidence_info is not None:
            metadata["evidence_type"] = self._get_text(evidence_info, "EvidenceType")
            metadata["examination_method"] = self._get_text(evidence_info, "ExaminationMethod")
        
        # Cadena de custodia
        custody_elem = metadata_elem.find("ChainOfCustody")
        if custody_elem is not None:
            metadata["chain_of_custody"] = []
            for entry in custody_elem.findall("CustodyEntry"):
                custody_entry = {}
                for child in entry:
                    custody_entry[child.tag] = child.text
                metadata["chain_of_custody"].append(custody_entry)
        
        return metadata
    
    def _parse_image_data(self, image_elem: ET.Element) -> Dict[str, Any]:
        """Parsea datos de imagen desde XML"""
        image_data = {}
        
        image_data["image_id"] = self._get_text(image_elem, "ImageID")
        image_data["file_path"] = self._get_text(image_elem, "FilePath")
        image_data["acquisition_date"] = self._get_text(image_elem, "AcquisitionDate")
        
        # Propiedades técnicas
        tech_props = image_elem.find("TechnicalProperties")
        if tech_props is not None:
            image_data["resolution_x"] = int(self._get_text(tech_props, "ResolutionX", "0"))
            image_data["resolution_y"] = int(self._get_text(tech_props, "ResolutionY", "0"))
            image_data["bit_depth"] = int(self._get_text(tech_props, "BitDepth", "0"))
            image_data["magnification"] = float(self._get_text(tech_props, "Magnification", "0.0"))
            image_data["lighting_conditions"] = self._get_text(tech_props, "LightingConditions")
        
        # Configuración de cámara
        camera_elem = image_elem.find("CameraSettings")
        if camera_elem is not None:
            image_data["camera_settings"] = {}
            for child in camera_elem:
                image_data["camera_settings"][child.tag] = child.text
        
        # Datos de calibración
        calibration_elem = image_elem.find("CalibrationData")
        if calibration_elem is not None:
            image_data["calibration_data"] = {}
            for child in calibration_elem:
                try:
                    image_data["calibration_data"][child.tag] = float(child.text)
                except (ValueError, TypeError):
                    image_data["calibration_data"][child.tag] = child.text
        
        return image_data
    
    def _parse_features(self, features_elem: ET.Element) -> List[Dict[str, Any]]:
        """Parsea características desde XML"""
        features = []
        
        for feature_elem in features_elem.findall("Feature"):
            feature = {}
            feature["feature_id"] = self._get_text(feature_elem, "FeatureID")
            feature["feature_type"] = self._get_text(feature_elem, "FeatureType")
            feature["extraction_method"] = self._get_text(feature_elem, "ExtractionMethod")
            feature["confidence_score"] = float(self._get_text(feature_elem, "ConfidenceScore", "0.0"))
            
            # Coordenadas
            coords_elem = feature_elem.find("Coordinates")
            if coords_elem is not None:
                feature["coordinates"] = []
                for point in coords_elem.findall("Point"):
                    coord = {}
                    for child in point:
                        try:
                            coord[child.tag] = float(child.text)
                        except (ValueError, TypeError):
                            coord[child.tag] = child.text
                    feature["coordinates"].append(coord)
            
            # Descriptores
            descriptors_elem = feature_elem.find("Descriptors")
            if descriptors_elem is not None:
                feature["descriptors"] = []
                for desc in descriptors_elem.findall("Descriptor"):
                    try:
                        feature["descriptors"].append(float(desc.text))
                    except (ValueError, TypeError):
                        feature["descriptors"].append(0.0)
            
            # Parámetros
            params_elem = feature_elem.find("Parameters")
            if params_elem is not None:
                feature["parameters"] = {}
                for child in params_elem:
                    feature["parameters"][child.tag] = child.text
            
            features.append(feature)
        
        return features
    
    def _parse_comparison_results(self, results_elem: ET.Element) -> List[Dict[str, Any]]:
        """Parsea resultados de comparación desde XML"""
        results = []
        
        for result_elem in results_elem.findall("ComparisonResult"):
            result = {}
            result["comparison_id"] = self._get_text(result_elem, "ComparisonID")
            result["reference_evidence_id"] = self._get_text(result_elem, "ReferenceEvidenceID")
            result["questioned_evidence_id"] = self._get_text(result_elem, "QuestionedEvidenceID")
            result["similarity_score"] = float(self._get_text(result_elem, "SimilarityScore", "0.0"))
            result["conclusion"] = self._get_text(result_elem, "Conclusion")
            result["confidence_level"] = self._get_text(result_elem, "ConfidenceLevel")
            result["examiner_notes"] = self._get_text(result_elem, "ExaminerNotes")
            
            # Puntos de coincidencia
            matches_elem = result_elem.find("MatchPoints")
            if matches_elem is not None:
                result["match_points"] = []
                for match in matches_elem.findall("MatchPoint"):
                    match_point = {}
                    for child in match:
                        try:
                            match_point[child.tag] = float(child.text)
                        except (ValueError, TypeError):
                            match_point[child.tag] = child.text
                    result["match_points"].append(match_point)
            
            results.append(result)
        
        return results
    
    def _get_text(self, parent: ET.Element, tag: str, default: str = "") -> str:
        """Obtiene texto de un elemento hijo"""
        elem = parent.find(tag)
        return elem.text if elem is not None and elem.text is not None else default
    
    def import_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        Importa datos desde archivo JSON
        
        Args:
            json_path: Ruta del archivo JSON
            
        Returns:
            Dict: Datos importados
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# Funciones de utilidad para crear objetos NIST

def create_nist_metadata(case_id: str, evidence_id: str, examiner_id: str, 
                        laboratory_id: str, evidence_type: str = "unknown",
                        examination_method: str = "digital_imaging") -> NISTMetadata:
    """
    Crea metadatos NIST con valores por defecto
    
    Args:
        case_id: ID del caso
        evidence_id: ID de la evidencia
        examiner_id: ID del examinador
        laboratory_id: ID del laboratorio
        evidence_type: Tipo de evidencia
        examination_method: Método de examinación
        
    Returns:
        NISTMetadata: Objeto de metadatos
    """
    now = datetime.now().isoformat()
    
    return NISTMetadata(
        case_id=case_id,
        evidence_id=evidence_id,
        examiner_id=examiner_id,
        laboratory_id=laboratory_id,
        examination_date=now,
        submission_date=now,
        evidence_type=EvidenceType(evidence_type),
        examination_method=ExaminationMethod(examination_method),
        chain_of_custody=[{
            "timestamp": now,
            "handler": examiner_id,
            "action": "initial_examination"
        }]
    )


def create_nist_image_data(image_id: str, file_path: str, 
                          resolution_x: int = 1920, resolution_y: int = 1080,
                          bit_depth: int = 8, magnification: float = 1.0) -> NISTImageData:
    """
    Crea datos de imagen NIST con valores por defecto
    
    Args:
        image_id: ID de la imagen
        file_path: Ruta del archivo
        resolution_x: Resolución horizontal
        resolution_y: Resolución vertical
        bit_depth: Profundidad de bits
        magnification: Magnificación
        
    Returns:
        NISTImageData: Objeto de datos de imagen
    """
    return NISTImageData(
        image_id=image_id,
        file_path=file_path,
        acquisition_date=datetime.now().isoformat(),
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        bit_depth=bit_depth,
        magnification=magnification,
        lighting_conditions="controlled",
        camera_settings={
            "iso": "100",
            "aperture": "f/8",
            "shutter_speed": "1/60",
            "white_balance": "auto"
        },
        calibration_data={
            "pixels_per_mm": 100.0,
            "scale_factor": 1.0,
            "rotation_angle": 0.0
        }
    )