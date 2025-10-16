"""
Comparison-specific widgets for the ballistic comparison tab.
These widgets provide specialized functionality for ballistic comparison workflows.
"""

from typing import Dict, Any, List, Optional, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QGroupBox, QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QProgressBar, QScrollArea, QTabWidget,
    QSplitter, QGridLayout, QButtonGroup, QRadioButton, QSlider,
    QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QPen

from .shared.stepper_widget import StepperWidget
try:
    from gui.shared_widgets import (
        NISTStandardsWidget, AFTEAnalysisWidget, DeepLearningWidget,
        ImageProcessingWidget
    )
except ImportError:
    # Fallback widgets
    from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
    
    class NISTStandardsWidget(QWidget):
        configuration_changed = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("NIST Standards Widget (Fallback)"))
    
    class AFTEAnalysisWidget(QWidget):
        configuration_changed = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("AFTE Analysis Widget (Fallback)"))
    
    class DeepLearningWidget(QWidget):
        configuration_changed = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Deep Learning Widget (Fallback)"))
    
    class ImageProcessingWidget(QWidget):
        configuration_changed = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Image Processing Widget (Fallback)"))


class ComparisonStepper(StepperWidget):
    """
    Specialized stepper widget for ballistic comparison workflow
    """
    
    def __init__(self, parent=None):
        # Define comparison-specific steps
        steps = [
            {
                'id': 'image_selection',
                'title': 'Image Selection',
                'description': 'Select evidence images for comparison',
                'icon': '📷'
            },
            {
                'id': 'comparison_mode',
                'title': 'Comparison Mode',
                'description': 'Choose direct comparison or database search',
                'icon': '⚖️'
            },
            {
                'id': 'configuration',
                'title': 'Configuration',
                'description': 'Configure comparison parameters',
                'icon': '⚙️'
            },
            {
                'id': 'comparison',
                'title': 'Comparison',
                'description': 'Execute ballistic comparison',
                'icon': '🔍'
            },
            {
                'id': 'results',
                'title': 'Results',
                'description': 'Review comparison results',
                'icon': '📊'
            }
        ]
        
        super().__init__(steps, parent)
        
        # Add comparison-specific navigation buttons
        self.setup_comparison_navigation()
    
    def setup_comparison_navigation(self):
        """Setup comparison-specific navigation buttons"""
        nav_layout = QHBoxLayout()
        
        # Mode switch button
        self.mode_switch_btn = QPushButton("Switch Mode")
        self.mode_switch_btn.setProperty("class", "secondary-button")
        self.mode_switch_btn.setToolTip("Switch between direct comparison and database search")
        nav_layout.addWidget(self.mode_switch_btn)
        
        nav_layout.addStretch()
        
        # Reset comparison button
        self.reset_comparison_btn = QPushButton("Reset Comparison")
        self.reset_comparison_btn.setProperty("class", "danger-button")
        self.reset_comparison_btn.setToolTip("Reset all comparison settings and results")
        nav_layout.addWidget(self.reset_comparison_btn)
        
        # Add to widget's layout
        self.layout().addLayout(nav_layout)
    
    def update_step_for_mode(self, mode: str):
        """Update step descriptions based on comparison mode"""
        if mode == 'direct':
            self.update_step_description('comparison_mode', 'Direct comparison between two evidence images')
            self.update_step_description('comparison', 'Execute direct ballistic comparison')
        elif mode == 'database':
            self.update_step_description('comparison_mode', 'Search database for similar evidence')
            self.update_step_description('comparison', 'Search ballistic database')


class ComparisonModeSelector(QWidget):
    """
    Widget for selecting comparison mode (direct vs database search)
    """
    
    mode_changed = pyqtSignal(str)  # 'direct' or 'database'
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mode = 'direct'
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the mode selector UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Comparison Mode")
        title.setProperty("class", "subtitle")
        layout.addWidget(title)
        
        # Mode selection
        mode_frame = QFrame()
        mode_frame.setProperty("class", "card")
        mode_layout = QVBoxLayout(mode_frame)
        
        # Button group for exclusive selection
        self.mode_group = QButtonGroup(self)
        
        # Direct comparison mode
        self.direct_radio = QRadioButton("Direct Comparison")
        self.direct_radio.setChecked(True)
        self.direct_radio.setToolTip("Compare two specific evidence images directly")
        self.mode_group.addButton(self.direct_radio, 0)
        
        direct_desc = QLabel("Compare two evidence images side by side with detailed analysis")
        direct_desc.setProperty("class", "caption")
        direct_desc.setWordWrap(True)
        
        # Database search mode
        self.database_radio = QRadioButton("Database Search")
        self.database_radio.setToolTip("Search database for similar evidence")
        self.mode_group.addButton(self.database_radio, 1)
        
        database_desc = QLabel("Search the ballistic database for evidence similar to your query image")
        database_desc.setProperty("class", "caption")
        database_desc.setWordWrap(True)
        
        # Add to layout
        mode_layout.addWidget(self.direct_radio)
        mode_layout.addWidget(direct_desc)
        mode_layout.addSpacing(10)
        mode_layout.addWidget(self.database_radio)
        mode_layout.addWidget(database_desc)
        
        layout.addWidget(mode_frame)
        
        # Connect signals
        self.mode_group.buttonClicked.connect(self.on_mode_changed)
    
    def on_mode_changed(self, button):
        """Handle mode change"""
        if button == self.direct_radio:
            self.current_mode = 'direct'
        else:
            self.current_mode = 'database'
        
        self.mode_changed.emit(self.current_mode)
    
    def get_current_mode(self) -> str:
        """Get the currently selected mode"""
        return self.current_mode
    
    def set_mode(self, mode: str):
        """Set the comparison mode programmatically"""
        if mode == 'direct':
            self.direct_radio.setChecked(True)
        elif mode == 'database':
            self.database_radio.setChecked(True)
        
        self.current_mode = mode
        self.mode_changed.emit(mode)


class ComparisonConfigManager(QWidget):
    """
    Hierarchical configuration manager for comparison settings
    """
    
    configuration_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mode = 'direct'
        self.current_level = 'Basic'
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the configuration manager UI"""
        layout = QVBoxLayout(self)
        
        # Configuration level selector
        level_frame = QFrame()
        level_frame.setProperty("class", "card")
        level_layout = QVBoxLayout(level_frame)
        
        level_title = QLabel("Configuration Level")
        level_title.setProperty("class", "subtitle")
        level_layout.addWidget(level_title)
        
        self.level_combo = QComboBox()
        self.level_combo.addItems(['Basic', 'Intermediate', 'Advanced'])
        self.level_combo.currentTextChanged.connect(self.on_level_changed)
        level_layout.addWidget(self.level_combo)
        
        layout.addWidget(level_frame)
        
        # Configuration tabs
        self.config_tabs = QTabWidget()
        
        # Image Processing tab
        self.image_processing_widget = ImageProcessingWidget()
        self.config_tabs.addTab(self.image_processing_widget, "Image Processing")
        
        # NIST Standards tab
        self.nist_widget = NISTStandardsWidget()
        self.config_tabs.addTab(self.nist_widget, "NIST Standards")
        
        # AFTE Analysis tab
        self.afte_widget = AFTEAnalysisWidget()
        self.config_tabs.addTab(self.afte_widget, "AFTE Analysis")
        
        # Deep Learning tab
        self.deep_learning_widget = DeepLearningWidget()
        self.config_tabs.addTab(self.deep_learning_widget, "Deep Learning")
        
        # Comparison-specific settings tab
        self.comparison_settings_widget = self.create_comparison_settings_widget()
        self.config_tabs.addTab(self.comparison_settings_widget, "Comparison Settings")
        
        layout.addWidget(self.config_tabs)
        
        # Configuration summary
        self.summary_widget = self.create_summary_widget()
        layout.addWidget(self.summary_widget)
        
        # Connect signals
        self.setup_connections()
    
    def create_comparison_settings_widget(self) -> QWidget:
        """Create comparison-specific settings widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Similarity threshold
        threshold_group = QGroupBox("Similarity Threshold")
        threshold_layout = QFormLayout(threshold_group)
        
        self.similarity_threshold = QDoubleSpinBox()
        self.similarity_threshold.setRange(0.0, 1.0)
        self.similarity_threshold.setSingleStep(0.01)
        self.similarity_threshold.setValue(0.75)
        self.similarity_threshold.setDecimals(2)
        threshold_layout.addRow("Minimum Similarity:", self.similarity_threshold)
        
        # Match criteria
        criteria_group = QGroupBox("Match Criteria")
        criteria_layout = QVBoxLayout(criteria_group)
        
        self.use_geometric_matching = QCheckBox("Geometric Feature Matching")
        self.use_geometric_matching.setChecked(True)
        criteria_layout.addWidget(self.use_geometric_matching)
        
        self.use_texture_analysis = QCheckBox("Texture Analysis")
        self.use_texture_analysis.setChecked(True)
        criteria_layout.addWidget(self.use_texture_analysis)
        
        self.use_statistical_correlation = QCheckBox("Statistical Correlation")
        self.use_statistical_correlation.setChecked(False)
        criteria_layout.addWidget(self.use_statistical_correlation)
        
        # Database search settings (only visible in database mode)
        self.db_settings_group = QGroupBox("Database Search Settings")
        db_layout = QFormLayout(self.db_settings_group)
        
        self.max_results = QSpinBox()
        self.max_results.setRange(1, 100)
        self.max_results.setValue(10)
        db_layout.addRow("Maximum Results:", self.max_results)
        
        self.search_radius = QDoubleSpinBox()
        self.search_radius.setRange(0.1, 2.0)
        self.search_radius.setSingleStep(0.1)
        self.search_radius.setValue(0.5)
        db_layout.addRow("Search Radius:", self.search_radius)
        
        # Add to main layout
        layout.addWidget(threshold_group)
        layout.addWidget(criteria_group)
        layout.addWidget(self.db_settings_group)
        
        return widget
    
    def create_summary_widget(self) -> QWidget:
        """Create configuration summary widget"""
        widget = QFrame()
        widget.setProperty("class", "card")
        layout = QVBoxLayout(widget)
        
        title = QLabel("Configuration Summary")
        title.setProperty("class", "subtitle")
        layout.addWidget(title)
        
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(100)
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        
        return widget
    
    def setup_connections(self):
        """Setup signal connections"""
        # Connect all configuration widgets
        self.image_processing_widget.configuration_changed.connect(self.on_config_changed)
        self.nist_widget.configuration_changed.connect(self.on_config_changed)
        self.afte_widget.configuration_changed.connect(self.on_config_changed)
        self.deep_learning_widget.configuration_changed.connect(self.on_config_changed)
        
        # Connect comparison-specific settings
        self.similarity_threshold.valueChanged.connect(self.on_config_changed)
        self.use_geometric_matching.toggled.connect(self.on_config_changed)
        self.use_texture_analysis.toggled.connect(self.on_config_changed)
        self.use_statistical_correlation.toggled.connect(self.on_config_changed)
        self.max_results.valueChanged.connect(self.on_config_changed)
        self.search_radius.valueChanged.connect(self.on_config_changed)
    
    def on_level_changed(self, level: str):
        """Handle configuration level change"""
        self.current_level = level
        self.update_widget_visibility()
        self.on_config_changed()
    
    def update_widget_visibility(self):
        """Update widget visibility based on configuration level"""
        # Update shared widgets
        self.image_processing_widget.set_configuration_level(self.current_level)
        self.nist_widget.set_configuration_level(self.current_level)
        self.afte_widget.set_configuration_level(self.current_level)
        self.deep_learning_widget.set_configuration_level(self.current_level)
        
        # Update comparison-specific settings visibility
        if self.current_level == 'Basic':
            self.use_statistical_correlation.setVisible(False)
            self.search_radius.setVisible(False)
        elif self.current_level == 'Intermediate':
            self.use_statistical_correlation.setVisible(True)
            self.search_radius.setVisible(False)
        else:  # Advanced
            self.use_statistical_correlation.setVisible(True)
            self.search_radius.setVisible(True)
    
    def set_comparison_mode(self, mode: str):
        """Set the comparison mode and update UI accordingly"""
        self.current_mode = mode
        
        # Show/hide database-specific settings
        self.db_settings_group.setVisible(mode == 'database')
        
        self.on_config_changed()
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the complete configuration"""
        config = {
            'level': self.current_level,
            'mode': self.current_mode,
            'image_processing': self.image_processing_widget.get_configuration(),
            'nist_standards': self.nist_widget.get_configuration(),
            'afte_analysis': self.afte_widget.get_configuration(),
            'deep_learning': self.deep_learning_widget.get_configuration(),
            'comparison_settings': {
                'similarity_threshold': self.similarity_threshold.value(),
                'use_geometric_matching': self.use_geometric_matching.isChecked(),
                'use_texture_analysis': self.use_texture_analysis.isChecked(),
                'use_statistical_correlation': self.use_statistical_correlation.isChecked(),
            }
        }
        
        if self.current_mode == 'database':
            config['database_settings'] = {
                'max_results': self.max_results.value(),
                'search_radius': self.search_radius.value()
            }
        
        return config
    
    def set_configuration(self, config: Dict[str, Any]):
        """Set the configuration from a dictionary"""
        if 'level' in config:
            self.level_combo.setCurrentText(config['level'])
        
        if 'mode' in config:
            self.set_comparison_mode(config['mode'])
        
        # Set shared widget configurations
        if 'image_processing' in config:
            self.image_processing_widget.set_configuration(config['image_processing'])
        
        if 'nist_standards' in config:
            self.nist_widget.set_configuration(config['nist_standards'])
        
        if 'afte_analysis' in config:
            self.afte_widget.set_configuration(config['afte_analysis'])
        
        if 'deep_learning' in config:
            self.deep_learning_widget.set_configuration(config['deep_learning'])
        
        # Set comparison-specific settings
        if 'comparison_settings' in config:
            cs = config['comparison_settings']
            self.similarity_threshold.setValue(cs.get('similarity_threshold', 0.75))
            self.use_geometric_matching.setChecked(cs.get('use_geometric_matching', True))
            self.use_texture_analysis.setChecked(cs.get('use_texture_analysis', True))
            self.use_statistical_correlation.setChecked(cs.get('use_statistical_correlation', False))
        
        if 'database_settings' in config:
            ds = config['database_settings']
            self.max_results.setValue(ds.get('max_results', 10))
            self.search_radius.setValue(ds.get('search_radius', 0.5))
    
    def on_config_changed(self):
        """Handle configuration changes"""
        config = self.get_configuration()
        self.update_summary()
        self.configuration_changed.emit(config)
    
    def update_summary(self):
        """Update the configuration summary"""
        config = self.get_configuration()
        
        summary_lines = [
            f"Configuration Level: {config['level']}",
            f"Comparison Mode: {config['mode'].title()}",
            f"Similarity Threshold: {config['comparison_settings']['similarity_threshold']:.2f}",
        ]
        
        if config['deep_learning']['enabled']:
            summary_lines.append(f"Deep Learning: {config['deep_learning']['model']}")
        
        if config['mode'] == 'database':
            summary_lines.append(f"Max Results: {config['database_settings']['max_results']}")
        
        self.summary_text.setPlainText('\n'.join(summary_lines))


class ComparisonProgressWidget(QWidget):
    """
    Widget for displaying comparison progress with detailed steps
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the progress widget UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Comparison Progress")
        title.setProperty("class", "subtitle")
        layout.addWidget(title)
        
        # Main progress bar
        self.main_progress = QProgressBar()
        self.main_progress.setTextVisible(True)
        layout.addWidget(self.main_progress)
        
        # Current step label
        self.current_step_label = QLabel("Ready to start comparison...")
        self.current_step_label.setProperty("class", "body")
        layout.addWidget(self.current_step_label)
        
        # Detailed progress area
        progress_frame = QFrame()
        progress_frame.setProperty("class", "card")
        progress_layout = QVBoxLayout(progress_frame)
        
        # Scrollable area for step details
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        
        self.progress_content = QWidget()
        self.progress_content_layout = QVBoxLayout(self.progress_content)
        scroll_area.setWidget(self.progress_content)
        
        progress_layout.addWidget(scroll_area)
        layout.addWidget(progress_frame)
    
    def update_progress(self, step: str, progress: int, details: str = ""):
        """Update the progress display"""
        self.main_progress.setValue(progress)
        self.current_step_label.setText(f"Current Step: {step}")
        
        if details:
            self.add_progress_detail(step, details)
    
    def add_progress_detail(self, step: str, details: str):
        """Add a detailed progress entry"""
        detail_label = QLabel(f"• {step}: {details}")
        detail_label.setProperty("class", "caption")
        detail_label.setWordWrap(True)
        self.progress_content_layout.addWidget(detail_label)
        
        # Auto-scroll to bottom
        QTimer.singleShot(100, lambda: self.scroll_to_bottom())
    
    def scroll_to_bottom(self):
        """Scroll the progress area to the bottom"""
        scroll_area = self.progress_content.parent().parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def reset_progress(self):
        """Reset the progress display"""
        self.main_progress.setValue(0)
        self.current_step_label.setText("Ready to start comparison...")
        
        # Clear progress details
        for i in reversed(range(self.progress_content_layout.count())):
            child = self.progress_content_layout.itemAt(i).widget()
            if child:
                child.setParent(None)


class ComparisonResultsWidget(QWidget):
    """
    Widget for displaying comparison results with detailed analysis
    """
    
    result_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_data = []
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the results widget UI"""
        layout = QVBoxLayout(self)
        
        # Title and controls
        header_layout = QHBoxLayout()
        
        title = QLabel("Comparison Results")
        title.setProperty("class", "subtitle")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setProperty("class", "secondary-button")
        header_layout.addWidget(self.export_btn)
        
        layout.addLayout(header_layout)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Summary tab
        self.summary_tab = self.create_summary_tab()
        self.results_tabs.addTab(self.summary_tab, "Summary")
        
        # Detailed results tab
        self.detailed_tab = self.create_detailed_tab()
        self.results_tabs.addTab(self.detailed_tab, "Detailed Results")
        
        # Visualization tab
        self.visualization_tab = self.create_visualization_tab()
        self.results_tabs.addTab(self.visualization_tab, "Visualization")
        
        layout.addWidget(self.results_tabs)
    
    def create_summary_tab(self) -> QWidget:
        """Create the summary results tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Summary cards
        self.summary_frame = QFrame()
        self.summary_layout = QGridLayout(self.summary_frame)
        layout.addWidget(self.summary_frame)
        
        layout.addStretch()
        return tab
    
    def create_detailed_tab(self) -> QWidget:
        """Create the detailed results tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.itemSelectionChanged.connect(self.on_result_selected)
        layout.addWidget(self.results_table)
        
        return tab
    
    def create_visualization_tab(self) -> QWidget:
        """Create the visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Visualization area
        self.visualization_area = QLabel("Visualization will appear here")
        self.visualization_area.setAlignment(Qt.AlignCenter)
        self.visualization_area.setMinimumHeight(300)
        self.visualization_area.setStyleSheet("border: 1px solid #ccc; background: #f9f9f9;")
        layout.addWidget(self.visualization_area)
        
        return tab
    
    def display_results(self, results: List[Dict[str, Any]]):
        """Display comparison results"""
        self.results_data = results
        
        # Update summary
        self.update_summary()
        
        # Update detailed table
        self.update_detailed_table()
        
        # Update visualization
        self.update_visualization()
    
    def update_summary(self):
        """Update the summary display"""
        # Clear existing summary
        for i in reversed(range(self.summary_layout.count())):
            child = self.summary_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        if not self.results_data:
            return
        
        # Create summary cards
        total_results = len(self.results_data)
        high_confidence = sum(1 for r in self.results_data if r.get('confidence', 0) > 0.8)
        avg_similarity = sum(r.get('similarity', 0) for r in self.results_data) / total_results if total_results > 0 else 0
        
        # Total results card
        total_card = self.create_summary_card("Total Results", str(total_results), "#3498db")
        self.summary_layout.addWidget(total_card, 0, 0)
        
        # High confidence card
        confidence_card = self.create_summary_card("High Confidence", str(high_confidence), "#27ae60")
        self.summary_layout.addWidget(confidence_card, 0, 1)
        
        # Average similarity card
        similarity_card = self.create_summary_card("Avg. Similarity", f"{avg_similarity:.2f}", "#e74c3c")
        self.summary_layout.addWidget(similarity_card, 0, 2)
    
    def create_summary_card(self, title: str, value: str, color: str) -> QWidget:
        """Create a summary card widget"""
        card = QFrame()
        card.setProperty("class", "card")
        card.setStyleSheet(f"QFrame {{ border-left: 4px solid {color}; }}")
        
        layout = QVBoxLayout(card)
        
        title_label = QLabel(title)
        title_label.setProperty("class", "caption")
        layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setProperty("class", "title")
        layout.addWidget(value_label)
        
        return card
    
    def update_detailed_table(self):
        """Update the detailed results table"""
        if not self.results_data:
            self.results_table.setRowCount(0)
            return
        
        # Setup table
        headers = ["ID", "Similarity", "Confidence", "AFTE Conclusion", "Match Type"]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setRowCount(len(self.results_data))
        
        # Populate table
        for row, result in enumerate(self.results_data):
            self.results_table.setItem(row, 0, QTableWidgetItem(str(result.get('id', row))))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{result.get('similarity', 0):.3f}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{result.get('confidence', 0):.3f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(result.get('afte_conclusion', 'Unknown')))
            self.results_table.setItem(row, 4, QTableWidgetItem(result.get('match_type', 'Unknown')))
        
        # Resize columns
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.resizeColumnsToContents()
    
    def update_visualization(self):
        """Update the visualization display"""
        # Create a simple visualization
        pixmap = QPixmap(400, 300)
        pixmap.fill(QColor(255, 255, 255))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.results_data:
            # Draw similarity distribution
            pen = QPen(QColor(52, 152, 219), 2)
            painter.setPen(pen)
            
            # Simple bar chart of similarities
            bar_width = 380 // len(self.results_data) if self.results_data else 1
            for i, result in enumerate(self.results_data[:10]):  # Show first 10 results
                similarity = result.get('similarity', 0)
                bar_height = int(similarity * 250)
                x = 10 + i * bar_width
                y = 280 - bar_height
                painter.drawRect(x, y, bar_width - 2, bar_height)
        
        painter.end()
        self.visualization_area.setPixmap(pixmap)
    
    def on_result_selected(self):
        """Handle result selection"""
        current_row = self.results_table.currentRow()
        if 0 <= current_row < len(self.results_data):
            selected_result = self.results_data[current_row]
            self.result_selected.emit(selected_result)
    
    def clear_results(self):
        """Clear all results"""
        self.results_data = []
        self.update_summary()
        self.update_detailed_table()
        self.update_visualization()