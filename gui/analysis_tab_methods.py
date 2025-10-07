"""
Métodos auxiliares reales para AnalysisTab
Estos métodos reemplazan las simulaciones con servicios reales del sistema
"""

def _initialize_processing_components(self):
    """Inicializa los componentes de procesamiento real del sistema"""
    try:
        self.logger.info("Inicializando componentes de procesamiento real")
        
        # Inicializar preprocesador unificado
        preprocessing_config = PreprocessingConfig(
            enable_noise_reduction=True,
            enable_contrast_enhancement=True,
            enable_sharpening=True,
            target_resolution=(1024, 1024)
        )
        self.preprocessor = UnifiedPreprocessor(preprocessing_config)
        
        # Inicializar detector de ROI
        roi_config = ROIDetectionConfig(
            detection_method='hybrid',
            confidence_threshold=0.7,
            min_roi_size=100
        )
        self.roi_detector = UnifiedROIDetector(roi_config)
        
        # Inicializar métricas de calidad NIST
        self.quality_metrics = NISTQualityMetrics()
        
        # Inicializar motor de conclusiones AFTE
        self.afte_engine = AFTEConclusionEngine()
        
        # Inicializar base de datos si está disponible
        try:
            self.database = UnifiedDatabase(self.config)
            self.logger.info("Base de datos inicializada correctamente")
        except Exception as e:
            self.logger.warning(f"No se pudo inicializar la base de datos: {e}")
            self.database = None
        
        self.logger.info("Componentes de procesamiento inicializados exitosamente")
        
    except Exception as e:
        self.logger.error(f"Error inicializando componentes de procesamiento: {e}")
        # Continuar sin fallar, pero registrar el error

def _validate_image_with_real_validator(self, image_path: str) -> tuple[bool, str]:
    """Valida la imagen usando el validador real del sistema"""
    try:
        return self.validator.validate_image_file(image_path)
    except Exception as e:
        self.logger.error(f"Error validando imagen: {e}")
        return False, f"Error de validación: {str(e)}"

def _get_real_processing_config(self) -> dict:
    """Obtiene la configuración real de procesamiento basada en la UI"""
    config = {
        'preprocessing': {
            'noise_reduction': getattr(self, 'noise_reduction_check', QCheckBox()).isChecked(),
            'contrast_enhancement': getattr(self, 'contrast_enhancement_check', QCheckBox()).isChecked(),
            'sharpening': getattr(self, 'sharpening_check', QCheckBox()).isChecked(),
            'target_resolution': (1024, 1024)
        },
        'roi_detection': {
            'method': 'hybrid',
            'confidence_threshold': getattr(self, 'roi_confidence_slider', QSlider()).value() / 100.0,
            'min_roi_size': 100
        },
        'feature_extraction': {
            'algorithm': 'unified',
            'enable_deep_learning': getattr(self, 'enable_dl_check', QCheckBox()).isChecked()
        },
        'quality_assessment': {
            'enable_nist_metrics': True,
            'quality_threshold': 0.7
        },
        'afte_analysis': {
            'enable_conclusions': True,
            'confidence_threshold': 0.6
        }
    }
    
    return config

def on_image_loaded_real(self, image_path: str):
    """Maneja la carga de imagen con validación real"""
    try:
        # Validar imagen con el validador real
        is_valid, message = self._validate_image_with_real_validator(image_path)
        
        if not is_valid:
            QMessageBox.warning(self, "Imagen no válida", message)
            return
        
        # Obtener información real de la imagen
        file_info = FileUtils.get_file_info(image_path)
        image_hash = SecurityUtils.calculate_file_hash(image_path)
        
        # Actualizar información en la UI
        self.image_name_label.setText(Path(image_path).name)
        self.image_size_label.setText(f"{file_info.get('size_mb', 0):.2f} MB")
        
        # Obtener dimensiones reales de la imagen
        import cv2
        image = cv2.imread(image_path)
        if image is not None:
            height, width = image.shape[:2]
            self.image_dimensions_label.setText(f"{width} x {height} px")
            self.image_format_label.setText(Path(image_path).suffix.upper())
        
        # Guardar datos de la imagen
        self.analysis_data['image_path'] = image_path
        self.analysis_data['image_hash'] = image_hash
        self.analysis_data['image_info'] = file_info
        
        # Mostrar información y habilitar siguiente paso
        self.image_info_frame.show()
        self.step2_group.setEnabled(True)
        
        self.logger.info(f"Imagen cargada y validada: {image_path}")
        
    except Exception as e:
        self.logger.error(f"Error cargando imagen: {e}")
        QMessageBox.critical(self, "Error", f"Error cargando imagen: {str(e)}")

def collect_real_analysis_data(self) -> dict:
    """Recolecta datos reales de análisis de la UI"""
    try:
        # Datos básicos
        analysis_data = {
            'image_path': self.analysis_data.get('image_path'),
            'image_hash': self.analysis_data.get('image_hash'),
            'evidence_type': self._get_evidence_type_mapping(),
            'config_level': self._get_configuration_level(),
            'enable_deep_learning': getattr(self, 'enable_dl_check', QCheckBox()).isChecked(),
            'save_to_database': True
        }
        
        # Datos del caso
        case_data = {
            'case_number': getattr(self, 'case_number_edit', QLineEdit()).text(),
            'evidence_id': getattr(self, 'evidence_id_edit', QLineEdit()).text(),
            'investigator': getattr(self, 'examiner_edit', QLineEdit()).text(),
            'date_created': getattr(self, 'acquisition_date_edit', QLineEdit()).text(),
            'weapon_type': getattr(self, 'weapon_make_edit', QLineEdit()).text(),
            'weapon_model': getattr(self, 'weapon_model_edit', QLineEdit()).text(),
            'caliber': getattr(self, 'caliber_edit', QLineEdit()).text(),
            'description': getattr(self, 'case_description_edit', QTextEdit()).toPlainText()
        }
        analysis_data['case_data'] = case_data
        
        # Metadatos NIST
        nist_metadata = {}
        if getattr(self, 'enable_nist_check', QCheckBox()).isChecked():
            nist_metadata = {
                'laboratory': getattr(self, 'laboratory_edit', QLineEdit()).text(),
                'equipment': getattr(self, 'equipment_edit', QLineEdit()).text(),
                'magnification': getattr(self, 'magnification_spin', QSpinBox()).value(),
                'lighting': getattr(self, 'lighting_combo', QComboBox()).currentText(),
                'operator': getattr(self, 'operator_edit', QLineEdit()).text()
            }
        analysis_data['nist_metadata'] = nist_metadata
        
        # Configuración de procesamiento real
        analysis_data['ballistic_config'] = self._get_real_processing_config()
        
        return analysis_data
        
    except Exception as e:
        self.logger.error(f"Error recolectando datos de análisis: {e}")
        return {}

def _get_evidence_type_mapping(self) -> str:
    """Mapea el tipo de evidencia de la UI al formato del sistema"""
    combo_text = getattr(self, 'evidence_type_combo', QComboBox()).currentText()
    
    mapping = {
        "Casquillo/Vaina (Cartridge Case)": "cartridge_case",
        "Bala/Proyectil (Bullet)": "bullet",
        "Proyectil General": "projectile"
    }
    
    return mapping.get(combo_text, "cartridge_case")

def _get_configuration_level(self) -> 'ConfigurationLevel':
    """Obtiene el nivel de configuración basado en la UI"""
    if getattr(self, 'high_precision_radio', QRadioButton()).isChecked():
        return ConfigurationLevel.HIGH_PRECISION
    elif getattr(self, 'research_radio', QRadioButton()).isChecked():
        return ConfigurationLevel.RESEARCH
    else:
        return ConfigurationLevel.STANDARD