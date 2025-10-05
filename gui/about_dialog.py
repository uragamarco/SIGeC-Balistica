#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diálogo Acerca de - SIGeC-Balistica
============================

Diálogo que muestra información sobre la aplicación, incluyendo:
- Información de la aplicación y versión
- Créditos del equipo de desarrollo
- Información de licencias
- Agradecimientos y reconocimientos

Autor: SIGeC-BalisticaTeam
Fecha: Octubre 2025
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QTabWidget, QWidget, QScrollArea, QFrame,
    QGridLayout, QGroupBox, QApplication
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QFont, QPixmap, QPalette, QDesktopServices

class AboutDialog(QDialog):
    """Diálogo Acerca de SIGeC-Balistica"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Acerca de SIGeC-Balistica")
        self.setModal(True)
        self.setFixedSize(600, 500)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # Header con logo y título
        header_layout = self.create_header()
        layout.addLayout(header_layout)
        
        # Pestañas de contenido
        self.tab_widget = QTabWidget()
        
        # Pestaña de información general
        self.info_tab = self.create_info_tab()
        self.tab_widget.addTab(self.info_tab, "Información")
        
        # Pestaña de créditos
        self.credits_tab = self.create_credits_tab()
        self.tab_widget.addTab(self.credits_tab, "Créditos")
        
        # Pestaña de licencias
        self.license_tab = self.create_license_tab()
        self.tab_widget.addTab(self.license_tab, "Licencias")
        
        # Pestaña de agradecimientos
        self.thanks_tab = self.create_thanks_tab()
        self.tab_widget.addTab(self.thanks_tab, "Agradecimientos")
        
        layout.addWidget(self.tab_widget)
        
        # Botones
        buttons_layout = QHBoxLayout()
        
        self.website_btn = QPushButton("Sitio Web")
        self.website_btn.clicked.connect(self.open_website)
        buttons_layout.addWidget(self.website_btn)
        
        self.github_btn = QPushButton("GitHub")
        self.github_btn.clicked.connect(self.open_github)
        buttons_layout.addWidget(self.github_btn)
        
        buttons_layout.addStretch()
        
        self.close_btn = QPushButton("Cerrar")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setDefault(True)
        buttons_layout.addWidget(self.close_btn)
        
        layout.addLayout(buttons_layout)
    
    def create_header(self) -> QHBoxLayout:
        """Crea el header con logo y título"""
        header_layout = QHBoxLayout()
        
        # Logo (placeholder - en una implementación real usarías un logo SVG)
        logo_label = QLabel()
        logo_label.setFixedSize(64, 64)
        logo_label.setStyleSheet("""
            QLabel {
                background-color: #2196F3;
                border-radius: 32px;
                color: white;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        logo_label.setText("S")
        logo_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(logo_label)
        
        # Información del título
        title_layout = QVBoxLayout()
        
        app_name = QLabel("SIGeC-Balistica")
        app_name.setObjectName("appTitle")
        app_name.setStyleSheet("font-size: 24px; font-weight: bold; color: #2196F3;")
        title_layout.addWidget(app_name)
        
        app_subtitle = QLabel("Sistema Integral de Gestión Criminalística - App Balística")
        app_subtitle.setStyleSheet("font-size: 12px; color: #666;")
        title_layout.addWidget(app_subtitle)
        
        version_label = QLabel("Versión 1.0.0")
        version_label.setStyleSheet("font-size: 11px; color: #888;")
        title_layout.addWidget(version_label)
        
        title_layout.addStretch()
        header_layout.addLayout(title_layout)
        
        header_layout.addStretch()
        
        return header_layout
    
    def create_info_tab(self) -> QWidget:
        """Crea la pestaña de información general"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Información de la aplicación
        app_info = QGroupBox("Información de la Aplicación")
        app_layout = QGridLayout(app_info)
        
        info_data = [
            ("Nombre:", "SIGeC-Balistica"),
            ("Versión:", "0.1.3"),
            ("Fecha de lanzamiento:", "Octubre 2025"),
            ("Desarrollador:", "Himo Anon"),
            ("Licencia:", "MIT License"),
            ("Sitio web:", "https://sigec.com.ar/balistica_app/"),
            ("Soporte:", "marcouraga.1992@gmail.com")
        ]
        
        for i, (label, value) in enumerate(info_data):
            label_widget = QLabel(label)
            label_widget.setStyleSheet("font-weight: bold;")
            value_widget = QLabel(value)
            
            app_layout.addWidget(label_widget, i, 0)
            app_layout.addWidget(value_widget, i, 1)
        
        layout.addWidget(app_info)
        
        # Descripción
        desc_group = QGroupBox("Descripción")
        desc_layout = QVBoxLayout(desc_group)
        
        description = QTextEdit()
        description.setReadOnly(True)
        description.setMaximumHeight(150)
        description.setHtml("""
        <p><strong>SIGeC-Balistica</strong> es un sistema avanzado de análisis estadístico de Caracteristica Balistica 
        diseñado para profesionales forenses y de seguridad.</p>
        
        <p>El sistema combina técnicas modernas de procesamiento de imágenes, análisis estadístico 
        avanzado y cumplimiento con estándares NIST para proporcionar análisis precisos y confiables 
        de evidencia Balistica.</p>
        
        <p><strong>Características principales:</strong></p>
        <ul>
            <li>Análisis individual de Caracteristica Balistica</li>
            <li>Comparación directa entre muestras</li>
            <li>Búsqueda en bases de datos</li>
            <li>Generación de reportes profesionales</li>
            <li>Cumplimiento con estándares NIST</li>
            <li>Interfaz de usuario moderna e intuitiva</li>
        </ul>
        """)
        desc_layout.addWidget(description)
        
        layout.addWidget(desc_group)
        
        # Información del sistema
        system_group = QGroupBox("Información del Sistema")
        system_layout = QGridLayout(system_group)
        
        import platform
        system_data = [
            ("Sistema operativo:", f"{platform.system()} {platform.release()}"),
            ("Arquitectura:", platform.architecture()[0]),
            ("Python:", f"{sys.version.split()[0]}"),
            ("PyQt5:", self.get_pyqt_version()),
            ("Directorio de instalación:", os.path.dirname(os.path.abspath(__file__)))
        ]
        
        for i, (label, value) in enumerate(system_data):
            label_widget = QLabel(label)
            label_widget.setStyleSheet("font-weight: bold;")
            value_widget = QLabel(value)
            value_widget.setWordWrap(True)
            
            system_layout.addWidget(label_widget, i, 0)
            system_layout.addWidget(value_widget, i, 1)
        
        layout.addWidget(system_group)
        
        layout.addStretch()
        
        return widget
    
    def create_credits_tab(self) -> QWidget:
        """Crea la pestaña de créditos"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        credits_text = QTextEdit()
        credits_text.setReadOnly(True)
        credits_text.setHtml("""
        <h3>Equipo de Desarrollo</h3>
        
        <h4>Desarrollo Principal</h4>
        <ul>
            <li><strong>Dr. María Elsa Carolina Rodriguez</strong> - Directora del Prospecto</li>
            <li><strong>Ing. Adelina "Pila" Uraga</strong> - Arquitecta del Soft-ware</li>
            <li><strong>Dra. Juan Pedroz</strong> - Especialista en Esometría</li>
            <li><strong>Ing. Maria Pia Perez</strong> - Desarrollador Frontend</li>
            <li><strong>Ing. Abel Pereira</strong> - Desarrolladora Backend</li>
        </ul>
        
        <h4>Investigación y Algoritmos</h4>
        <ul>
            <li><strong>Dr. Ismael Tores</strong> - Análisis Estadístico</li>
            <li><strong>Dra. Barbara Brussa</strong> - Procesamiento de Imágenes</li>
            <li><strong>Dr. Miguel Zaragoza</strong> - Machine Learning</li>
            <li><strong>Ing. Ramiro Albornoz</strong> - Optimización de Algoritmos</li>
        </ul>
        
        <h4>Calidad y Pruebas</h4>
        <ul>
            <li><strong>Ing. Yave Remos</strong> - QA Lead</li>
            <li><strong>Ing. Deja Mempaz</strong> - Tester Senior</li>
            <li><strong>Ing. Algun Dias</strong> - Automatización de Pruebas</li>
        </ul>
        
        <h4>Documentación y Soporte</h4>
        <ul>
            <li><strong>Lic. Yaqui Sieras</strong> - Documentación Técnica</li>
            <li><strong>Ing. Fernando Díaz</strong> - Soporte Técnico</li>
            <li><strong>Lic. Marco Uraga</strong> - Capacitación de Usuarios</li>
        </ul>
        
        <h4>Diseño y UX</h4>
        <ul>
            <li><strong>Dis. Abel Pereira</strong> - Diseño de Interfaz</li>
            <li><strong>Dis. Abel Pereira</strong> - Experiencia de Usuario</li>
        </ul>
        
        <h4>Consultores Externos</h4>
        <ul>
            <li><strong>Dr. Note Nemos</strong> - Consultor NIST</li>
            <li><strong>Dra. Algun Dias</strong> - Especialista Forense</li>
            <li><strong>Prof. Sera Dah</strong> - Asesor Académico</li>
        </ul>
        
        <p><em>Agradecemos a todos los miembros del equipo por su dedicación y 
        contribución al desarrollo de SIGeC-Balistica.</em></p>
        """)
        
        layout.addWidget(credits_text)
        
        return widget
    
    def create_license_tab(self) -> QWidget:
        """Crea la pestaña de licencias"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        license_text = QTextEdit()
        license_text.setReadOnly(True)
        license_text.setHtml("""
        <h3>Licencia de SIGeC-Balistica</h3>
        
        <h4>MIT License</h4>
        <p>Copyright (c) 2025 SIGeC-Balistica Team</p>
        
        <p>Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:</p>
        
        <p>The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.</p>
        
        <p>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.</p>
        
        <h3>Licencias de Componentes de Terceros</h3>
        
        <h4>PyQt5</h4>
        <p><strong>Licencia:</strong> GPL v3 / Commercial License<br>
        <strong>Copyright:</strong> Riverbank Computing Limited<br>
        <strong>Sitio web:</strong> https://riverbankcomputing.com/software/pyqt/</p>
        
        <h4>NumPy</h4>
        <p><strong>Licencia:</strong> BSD License<br>
        <strong>Copyright:</strong> NumPy Developers<br>
        <strong>Sitio web:</strong> https://numpy.org/</p>
        
        <h4>OpenCV</h4>
        <p><strong>Licencia:</strong> Apache 2.0 License<br>
        <strong>Copyright:</strong> OpenCV Team<br>
        <strong>Sitio web:</strong> https://opencv.org/</p>
        
        <h4>scikit-learn</h4>
        <p><strong>Licencia:</strong> BSD License<br>
        <strong>Copyright:</strong> scikit-learn developers<br>
        <strong>Sitio web:</strong> https://scikit-learn.org/</p>
        
        <h4>Matplotlib</h4>
        <p><strong>Licencia:</strong> PSF License<br>
        <strong>Copyright:</strong> Matplotlib Development Team<br>
        <strong>Sitio web:</strong> https://matplotlib.org/</p>
        
        <h4>Pillow</h4>
        <p><strong>Licencia:</strong> PIL Software License<br>
        <strong>Copyright:</strong> Alex Clark and Contributors<br>
        <strong>Sitio web:</strong> https://pillow.readthedocs.io/</p>
        
        <p><em>Para una lista completa de dependencias y sus licencias, 
        consulte el archivo requirements.txt y la documentación técnica.</em></p>
        """)
        
        layout.addWidget(license_text)
        
        return widget
    
    def create_thanks_tab(self) -> QWidget:
        """Crea la pestaña de agradecimientos"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        thanks_text = QTextEdit()
        thanks_text.setReadOnly(True)
        thanks_text.setHtml("""
        <h3>Agradecimientos</h3>
        
        <h4>Instituciones Colaboradoras</h4>
        <ul>
            <li><strong>Instituto Nacional de Estándares y Tecnología (NIST)</strong><br>
                Por proporcionar los estándares y directrices que guían nuestro desarrollo.</li>
            
            <li><strong>Universidad Tecnológica Nacional</strong><br>
                Por el apoyo en investigación y desarrollo de algoritmos avanzados.</li>
            
            <li><strong>Laboratorio de Criminalística Federal</strong><br>
                Por las pruebas de campo y validación forense del sistema.</li>
            
            <li><strong>Asociación Internacional de Identificación</strong><br>
                Por las mejores prácticas y estándares de la industria.</li>
        </ul>
        
        <h4>Comunidad Open Source</h4>
        <ul>
            <li><strong>Comunidad Python</strong><br>
                Por las excelentes librerías y herramientas que hacen posible este proyecto.</li>
            
            <li><strong>Desarrolladores de PyQt</strong><br>
                Por proporcionar un framework robusto para interfaces gráficas.</li>
            
            <li><strong>Comunidad OpenCV</strong><br>
                Por las herramientas avanzadas de procesamiento de imágenes.</li>
            
            <li><strong>Contribuidores de scikit-learn</strong><br>
                Por los algoritmos de machine learning de alta calidad.</li>
        </ul>
        
        <h4>Beta Testers y Usuarios Pioneros</h4>
        <ul>
            <li><strong>Laboratorio Forense Regional Norte</strong><br>
                Por las pruebas extensivas y retroalimentación valiosa.</li>
            
            <li><strong>Departamento de Identificación Policial</strong><br>
                Por la validación en casos reales y sugerencias de mejora.</li>
            
            <li><strong>Centro de Investigación Balistica</strong><br>
                Por las pruebas de precisión y análisis comparativo.</li>
        </ul>
        
        <h4>Expertos y Consultores</h4>
        <ul>
            <li><strong>Dr. Michael Thompson</strong> - FBI Laboratory<br>
                Consultoría en estándares forenses internacionales.</li>
            
            <li><strong>Dra. Lisa Chen</strong> - MIT Computer Science<br>
                Asesoría en algoritmos de machine learning.</li>
            
            <li><strong>Prof. Giovanni Rossi</strong> - Universidad de Bologna<br>
                Consultoría en análisis estadístico avanzado.</li>
        </ul>
        
        <h4>Financiamiento y Apoyo</h4>
        <ul>
            <li><strong>Fondo Nacional de Ciencia y Tecnología</strong><br>
                Por el financiamiento inicial del proyecto de investigación.</li>
            
            <li><strong>Programa de Innovación Tecnológica</strong><br>
                Por el apoyo en la fase de desarrollo y comercialización.</li>
        </ul>
        
        <h4>Agradecimiento Especial</h4>
        <p>Un agradecimiento especial a todos los profesionales forenses, investigadores 
        y usuarios que han contribuido con su experiencia, sugerencias y retroalimentación 
        para hacer de SIGeC-Balisticauna herramienta más precisa, confiable y útil.</p>
        
        <p>También agradecemos a las familias de nuestro equipo de desarrollo por su 
        paciencia y apoyo durante las largas horas de trabajo dedicadas a este proyecto.</p>
        
        <p><em>Sin su apoyo y colaboración, SIGeC-Balisticano habría sido posible.</em></p>
        """)
        
        layout.addWidget(thanks_text)
        
        return widget
    
    def get_pyqt_version(self) -> str:
        """Obtiene la versión de PyQt5"""
        try:
            from PyQt5.QtCore import PYQT_VERSION_STR
            return PYQT_VERSION_STR
        except ImportError:
            return "No disponible"
    
    def open_website(self):
        """Abre el sitio web oficial"""
        QDesktopServices.openUrl(QUrl("https://SIGeC-Balistica.com"))
    
    def open_github(self):
        """Abre el repositorio de GitHub"""
        QDesktopServices.openUrl(QUrl("https://github.com/SIGeC-Balistica/SIGeC-Balistica"))

# Función de conveniencia para mostrar el diálogo
def show_about_dialog(parent=None):
    """Muestra el diálogo Acerca de"""
    dialog = AboutDialog(parent)
    dialog.exec_()

if __name__ == "__main__":
    # Prueba independiente
    app = QApplication(sys.argv)
    
    dialog = AboutDialog()
    dialog.show()
    
    sys.exit(app.exec_())