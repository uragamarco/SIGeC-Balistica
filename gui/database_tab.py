"""
Modern Database Tab for Ballistic Analysis System
Refactored with hierarchical configuration and step-by-step navigation
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QScrollArea, QSplitter, QFrame, QSpacerItem,
    QSizePolicy, QFileDialog, QMessageBox, QProgressBar, QTabWidget,
    QListWidget, QListWidgetItem, QSlider, QDoubleSpinBox, QDateEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QTreeWidget, QTreeWidgetItem,
    QButtonGroup, QRadioButton, QCalendarWidget, QApplication, QMenu, QAction
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QDate, QSize
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPainter, QPen, QColor, QBrush
from .styles import apply_modern_qss_to_widget

# Telemetry integration
try:
    from core.telemetry_system import record_user_action, record_feature_usage, record_performance_event, record_error_event
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    # Mock functions for development
    def record_user_action(*args, **kwargs): pass
    def record_feature_usage(*args, **kwargs): pass
    def record_performance_event(*args, **kwargs): pass
    def record_error_event(*args, **kwargs): pass

# Cache system integration
try:
    from core.intelligent_cache import get_cache, initialize_cache
    from image_processing.lbp_cache import get_lbp_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    # Mock functions for development
    def get_cache(): return None
    def initialize_cache(*args, **kwargs): return None
    def get_lbp_cache(): return None

# Import shared widgets
from .shared_widgets import (
    ImageDropZone, ResultCard, CollapsiblePanel, StepIndicator, 
    ProgressCard, ImageViewer
)

# Import database-specific widgets
from .widgets.database import (
    DatabaseStepper, DatabaseSearchWidget, DatabaseResultsWidget,
    DatabaseDashboardWidget, DatabaseProgressWidget
)

# Import visualization widgets
from .visualization_widgets import VisualizationPanel
from .dynamic_results_panel import ResultsPanel

# Backend integration
from .backend_integration import get_backend_integration

logger = logging.getLogger(__name__)


class DatabaseWorker(QThread):
    """
    Background worker for database operations
    """
    
    # Signals
    progress_updated = pyqtSignal(int, str)
    search_completed = pyqtSignal(list)
    statistics_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    step_completed = pyqtSignal(str, dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.operation = None
        self.parameters = {}
        self.backend = get_backend_integration()
        self._is_cancelled = False
        self.start_time = None
        
        # Initialize cache systems
        self.cache = None
        self.lbp_cache = None
        if CACHE_AVAILABLE:
            try:
                self.cache = get_cache()
                self.lbp_cache = get_lbp_cache()
            except Exception as e:
                logger.warning(f"Could not initialize cache systems: {e}")
                self.cache = None
                self.lbp_cache = None
    
    def set_operation(self, operation: str, parameters: Dict[str, Any]):
        """Set the operation to perform"""
        self.operation = operation
        self.parameters = parameters
        self._is_cancelled = False
    
    def cancel(self):
        """Cancel the current operation"""
        self._is_cancelled = True
    
    def run(self):
        """Execute the database operation"""
        self.start_time = time.time()
        
        try:
            # Record operation start
            record_user_action(f'database_{self.operation}_started', 'database_tab', 
                             data={
                                 'operation': self.operation,
                                 'parameters': self.parameters
                             })
            
            if self.operation == "search":
                self._perform_search()
            elif self.operation == "statistics":
                self._load_statistics()
            elif self.operation == "export":
                self._export_data()
            else:
                self.error_occurred.emit(f"Unknown operation: {self.operation}")
        
        except Exception as e:
            # Record error
            record_error_event(e, 'database_worker', f'{self.operation}_operation')
            logger.error(f"Database operation error: {str(e)}")
            self.error_occurred.emit(f"Operation failed: {str(e)}")
    
    def _perform_search(self):
        """Perform database search with intelligent caching"""
        search_start = time.time()
        self.progress_updated.emit(10, "Initializing search...")
        
        # Generate cache key for search
        cache_key = None
        cache_hits = 0
        cache_misses = 0
        
        if self.cache:
            import hashlib
            cache_data = {
                'operation': 'search',
                'parameters': self.parameters
            }
            cache_key = hashlib.md5(str(cache_data).encode()).hexdigest()
            
            # Try to get cached result
            cached_result = self.cache.get(cache_key)
            if cached_result:
                cache_hits = 1
                self.progress_updated.emit(100, "Retrieved from cache!")
                
                # Add cache statistics to results
                if isinstance(cached_result, list) and cached_result:
                    for result in cached_result:
                        if isinstance(result, dict):
                            result['cache_stats'] = {
                                'hits': cache_hits,
                                'misses': cache_misses,
                                'hit_rate': 1.0
                            }
                
                self.search_completed.emit(cached_result)
                return
            else:
                cache_misses = 1
        
        if self._is_cancelled:
            return
        
        # Simulate search steps with performance tracking
        search_steps = [
            (20, "Connecting to database..."),
            (40, "Executing query..."),
            (60, "Processing results..."),
            (80, "Formatting data..."),
            (100, "Search completed")
        ]
        
        for progress, message in search_steps:
            if self._is_cancelled:
                return
            
            self.progress_updated.emit(progress, message)
            
            # Simulate processing time
            import time as time_module
            time_module.sleep(0.3)
        
        # Record search performance
        search_time = (time.time() - search_start) * 1000
        record_performance_event('database_search', 'database_worker', search_time, 
                               success=True, metadata={
                                   'search_params': self.parameters,
                                   'results_count': len(self.parameters.get('mock_results', [])),
                                   'cache_enabled': self.cache is not None
                               })
        
        # Mock results for demonstration
        mock_results = [
            {
                "case_id": "CASE-2024-001",
                "caliber": "9mm",
                "weapon_type": "Pistol",
                "date": "2024-01-15",
                "location": "Evidence Room A",
                "status": "Active"
            },
            {
                "case_id": "CASE-2024-002", 
                "caliber": ".45 ACP",
                "weapon_type": "Pistol",
                "date": "2024-01-20",
                "location": "Evidence Room B",
                "status": "Processed"
            }
        ]
        
        # Add cache statistics to results
        if self.cache:
            for result in mock_results:
                result['cache_stats'] = {
                    'hits': cache_hits,
                    'misses': cache_misses,
                    'hit_rate': cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
                }
            
            # Store results in cache with 30 minutes TTL
            try:
                self.cache.set(cache_key, mock_results, ttl=1800)
            except Exception as e:
                logger.warning(f"Could not cache search results: {e}")
        
        self.search_completed.emit(mock_results)
    
    def _load_statistics(self):
        """Load database statistics"""
        stats_start = time.time()
        self.progress_updated.emit(50, "Loading statistics...")
        
        # Simulate loading time
        import time as time_module
        time_module.sleep(0.5)
        
        # Record statistics loading performance
        stats_time = (time.time() - stats_start) * 1000
        record_performance_event('database_statistics', 'database_worker', stats_time, success=True)
        
        # Mock statistics
        mock_stats = {
            "total_cases": 1250,
            "active_cases": 89,
            "processed_cases": 1161,
            "calibers": {
                "9mm": 450,
                ".45 ACP": 320,
                ".40 S&W": 280,
                ".38 Special": 200
            }
        }
        
        self.statistics_updated.emit(mock_stats)
    
    def _export_data(self):
        """Export database results"""
        export_start = time.time()
        
        try:
            data = self.parameters.get("data", [])
            format_type = self.parameters.get("format", "JSON")
            file_path = self.parameters.get("file_path", "")
            
            self.progress_updated.emit(25, "Preparing export data...")
            
            # Simulate export processing
            import time as time_module
            time_module.sleep(0.8)
            
            self.progress_updated.emit(75, f"Exporting to {format_type}...")
            time_module.sleep(0.5)
            
            # Record export performance
            export_time = (time.time() - export_start) * 1000
            record_performance_event('database_export', 'database_worker', export_time, 
                                   success=True, metadata={
                                       'format': format_type,
                                       'record_count': len(data),
                                       'file_path': file_path
                                   })
            
            self.progress_updated.emit(100, "Export completed")
            self.step_completed.emit("export", {"file_path": file_path, "format": format_type})
            
        except Exception as e:
            record_error_event(e, 'database_worker', 'export_data')
            self.error_occurred.emit(f"Export failed: {str(e)}")


class DatabaseTab(QWidget):
    """
    Modern Database Tab with hierarchical configuration and step-by-step navigation
    """
    
    # Signals
    search_completed = pyqtSignal(list)
    item_selected = pyqtSignal(dict)
    export_requested = pyqtSignal(list, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State management
        self.current_results: List[Dict[str, Any]] = []
        self.selected_item: Optional[Dict[str, Any]] = None
        self.current_filters: Dict[str, Any] = {}
        self.database_worker: Optional[DatabaseWorker] = None
        
        # Initialize UI
        self.init_ui()
        self.setup_connections()
        self.apply_modern_theme()
        
        # Load initial statistics
        self.load_statistics()
        
        logger.info("Modern DatabaseTab initialized")
    
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        
        # Left panel - Navigation and configuration
        left_panel = self.create_left_panel()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)
        main_splitter.addWidget(left_panel)
        
        # Right panel - Content and results
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([350, 850])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(main_splitter)
        
        # Status bar
        status_bar = self.create_status_bar()
        main_layout.addWidget(status_bar)
    
    def create_header(self) -> QWidget:
        """Create the header section"""
        header = QFrame()
        header.setObjectName("headerFrame")
        header.setFixedHeight(60)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Title and description
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        
        title = QLabel("Ballistic Database")
        title.setObjectName("headerTitle")
        title_layout.addWidget(title)
        
        subtitle = QLabel("Search, analyze, and manage ballistic evidence")
        subtitle.setObjectName("headerSubtitle")
        title_layout.addWidget(subtitle)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        # Quick actions
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_database)
        actions_layout.addWidget(self.refresh_btn)
        
        self.settings_btn = QPushButton("⚙️ Settings")
        actions_layout.addWidget(self.settings_btn)
        
        layout.addLayout(actions_layout)
        
        return header
    
    def create_left_panel(self) -> QWidget:
        """Create the left navigation and configuration panel"""
        panel = QFrame()
        panel.setObjectName("leftPanel")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Database stepper
        self.stepper = DatabaseStepper()
        self.stepper.stepChanged.connect(self.on_step_changed)
        layout.addWidget(self.stepper)
        
        # Search configuration
        search_group = QGroupBox("Search Configuration")
        search_layout = QVBoxLayout(search_group)
        
        self.search_widget = DatabaseSearchWidget()
        self.search_widget.searchRequested.connect(self.perform_search)
        self.search_widget.filtersChanged.connect(self.on_filters_changed)
        search_layout.addWidget(self.search_widget)
        
        layout.addWidget(search_group)
        
        # Progress tracking
        progress_group = QGroupBox("Operation Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_widget = DatabaseProgressWidget()
        progress_layout.addWidget(self.progress_widget)
        
        layout.addWidget(progress_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right content panel"""
        panel = QFrame()
        panel.setObjectName("rightPanel")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Content tabs
        self.content_tabs = QTabWidget()
        
        # Dashboard tab
        dashboard_tab = self.create_dashboard_tab()
        self.content_tabs.addTab(dashboard_tab, "📊 Dashboard")
        
        # Search results tab
        results_tab = self.create_results_tab()
        self.content_tabs.addTab(results_tab, "🔍 Search Results")
        
        # Visualization tab
        visualization_tab = self.create_visualization_tab()
        self.content_tabs.addTab(visualization_tab, "📈 Visualization")
        
        layout.addWidget(self.content_tabs)
        
        return panel
    
    def create_dashboard_tab(self) -> QWidget:
        """Create the dashboard tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Dashboard widget
        self.dashboard_widget = DatabaseDashboardWidget()
        layout.addWidget(self.dashboard_widget)
        
        return tab
    
    def create_results_tab(self) -> QWidget:
        """Create the search results tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Results widget
        self.results_widget = DatabaseResultsWidget()
        self.results_widget.itemSelected.connect(self.on_item_selected)
        self.results_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.results_widget.exportRequested.connect(self.export_results)
        layout.addWidget(self.results_widget)
        
        return tab
    
    def create_visualization_tab(self) -> QWidget:
        """Create the visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Visualization panel
        self.visualization_panel = VisualizationPanel()
        layout.addWidget(self.visualization_panel)
        
        return tab
    
    def create_status_bar(self) -> QWidget:
        """Create the status bar"""
        status_bar = QFrame()
        status_bar.setObjectName("statusBar")
        status_bar.setFixedHeight(30)
        
        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(20, 5, 20, 5)
        
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Cache status indicator
        self.cache_status = QLabel("Cache: " + ("✓" if CACHE_AVAILABLE else "✗"))
        self.cache_status.setObjectName("systemStatus")
        layout.addWidget(self.cache_status)
        
        # Connection status
        self.connection_label = QLabel("🟢 Connected")
        self.connection_label.setObjectName("connectionLabel")
        layout.addWidget(self.connection_label)
        
        return status_bar
    
    def setup_connections(self):
        """Setup signal connections"""
        # Worker connections will be set up when worker is created
        pass
    
    def apply_modern_theme(self):
        """Apply modern theme styling"""
        # First apply sanitized global modern theme if available
        apply_modern_qss_to_widget(self)
        # Then apply specific inline styles for DatabaseTab
        self.setStyleSheet("""
            /* Header styling */
            QFrame#headerFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                border: none;
                border-radius: 8px;
            }
            
            QLabel#headerTitle {
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            
            QLabel#headerSubtitle {
                color: rgba(255, 255, 255, 0.8);
                font-size: 12px;
            }
            
            /* Panel styling */
            QFrame#leftPanel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
            
            QFrame#rightPanel {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
            
            /* Status bar styling */
            QFrame#statusBar {
                background-color: #f8f9fa;
                border-top: 1px solid #dee2e6;
            }
            
            QLabel#statusLabel {
                color: #6c757d;
                font-size: 11px;
            }
            
            QLabel#connectionLabel {
                color: #28a745;
                font-size: 11px;
                font-weight: bold;
            }
            
            /* Group box styling */
            QGroupBox {
                font-weight: bold;
                color: #495057;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #f8f9fa;
            }
            
            /* Button styling */
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: #0056b3;
            }
            
            QPushButton:pressed {
                background-color: #004085;
            }
            
            QPushButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }
            
            /* Tab styling */
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
            }
            
            QTabBar::tab {
                background-color: #f8f9fa;
                color: #495057;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: white;
                color: #007bff;
                border-bottom: 2px solid #007bff;
            }
            
            QTabBar::tab:hover {
                background-color: #e9ecef;
            }
        """)
    
    def on_step_changed(self, step: int):
        """Handle step change in the stepper"""
        # Record step navigation
        record_user_action('database_step_changed', 'database_tab', 
                         data={
                             'step': step,
                             'step_name': ["Search", "Filter", "Results", "Export"][step] if 0 <= step < 4 else "Unknown"
                         })
        
        step_names = ["Search", "Filter", "Results", "Export"]
        if 0 <= step < len(step_names):
            self.status_label.setText(f"Step {step + 1}: {step_names[step]}")
            
            # Switch to appropriate tab based on step
            if step == 0:  # Search
                self.content_tabs.setCurrentIndex(0)  # Dashboard
            elif step in [1, 2]:  # Filter, Results
                self.content_tabs.setCurrentIndex(1)  # Search Results
            elif step == 3:  # Export
                self.content_tabs.setCurrentIndex(2)  # Visualization

    def on_filters_changed(self, filters: Dict[str, Any]):
        """Handle filter changes"""
        # Record filter usage
        record_feature_usage('database_filters', 'database_tab', {
            'filters_applied': filters,
            'filter_count': len(filters)
        })
        
        self.current_filters = filters
        logger.debug(f"Filters updated: {filters}")
    
    def perform_search(self, search_params: Dict[str, Any]):
        """Perform database search"""
        # Record search initiation
        record_user_action('database_search_initiated', 'database_tab', 
                          data={
                              'search_params': search_params,
                              'filters': self.current_filters
                          })
        
        self.status_label.setText("Searching database...")
        self.results_widget.show_loading(True)
        
        # Create and start worker
        self.database_worker = DatabaseWorker()
        self.database_worker.progress_updated.connect(self.on_progress_updated)
        self.database_worker.search_completed.connect(self.on_search_completed)
        self.database_worker.error_occurred.connect(self.on_error_occurred)
        
        self.database_worker.set_operation("search", search_params)
        self.database_worker.start()
    
    def on_progress_updated(self, progress: int, message: str):
        """Handle progress updates"""
        self.progress_widget.set_progress(progress)
        self.progress_widget.set_status(message)
        self.progress_widget.add_detail(message)
    
    def on_search_completed(self, results: List[Dict[str, Any]]):
        """Handle search completion"""
        # Record search completion
        record_user_action('database_search_completed', 'database_tab',
                          data={
                              'results_count': len(results),
                              'search_successful': True
                          })
        
        self.current_results = results
        self.results_widget.set_results(results)
        self.results_widget.show_loading(False)
        
        self.status_label.setText(f"Found {len(results)} results")
        self.search_completed.emit(results)
        
        # Switch to results tab
        self.content_tabs.setCurrentIndex(1)
        
        # Move stepper to results step
        self.stepper.set_step(2)
    
    def on_error_occurred(self, error_message: str):
        """Handle errors"""
        # Record error occurrence
        record_user_action('database_error_occurred', 'database_tab',
                          data={
                              'error_message': error_message,
                              'operation': getattr(self.database_worker, 'operation', 'unknown') if self.database_worker else 'unknown'
                          })
        
        self.results_widget.show_loading(False)
        self.status_label.setText("Error occurred")
        
        QMessageBox.critical(self, "Database Error", error_message)
        logger.error(f"Database error: {error_message}")
    
    def on_item_selected(self, item: Dict[str, Any]):
        """Handle item selection"""
        # Record item selection
        record_user_action('database_item_selected', 'database_tab',
                          data={
                              'case_id': item.get('case_id', 'unknown'),
                              'caliber': item.get('caliber', 'unknown'),
                              'weapon_type': item.get('weapon_type', 'unknown')
                          })
        
        self.selected_item = item
        self.item_selected.emit(item)
        
        # Update visualization with selected item
        if hasattr(self.visualization_panel, 'set_data'):
            self.visualization_panel.set_data(item)
    
    def on_item_double_clicked(self, item: Dict[str, Any]):
        """Handle item double-click"""
        # Record item details view
        record_user_action('database_item_details_viewed', 'database_tab',
                          data={
                              'case_id': item.get('case_id', 'unknown')
                          })
        
        # Open detailed view or comparison
        self.show_item_details(item)
    
    def show_item_details(self, item: Dict[str, Any]):
        """Show detailed view of an item"""
        # Implementation would show a detailed dialog or panel
        QMessageBox.information(
            self, 
            "Item Details", 
            f"Case ID: {item.get('case_id', 'Unknown')}\n"
            f"Caliber: {item.get('caliber', 'Unknown')}\n"
            f"Weapon Type: {item.get('weapon_type', 'Unknown')}\n"
            f"Date: {item.get('date', 'Unknown')}"
        )
    
    def export_results(self, results: List[Dict[str, Any]], format_type: str):
        """Export search results"""
        if not results:
            QMessageBox.warning(self, "Export Warning", "No results to export")
            return
        
        # Record export initiation
        record_feature_usage('database_export', 'database_tab',
                            data={
                                'format': format_type,
                                'record_count': len(results)
                            })
        
        # Get export file path
        file_filter = {
            "JSON": "JSON files (*.json)",
            "CSV": "CSV files (*.csv)",
            "PDF": "PDF files (*.pdf)",
            "Excel": "Excel files (*.xlsx)"
        }.get(format_type, "All files (*.*)")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Export {format_type}", f"database_results.{format_type.lower()}", file_filter
        )
        
        if file_path:
            # Create and start export worker
            self.database_worker = DatabaseWorker()
            self.database_worker.progress_updated.connect(self.on_progress_updated)
            self.database_worker.step_completed.connect(self.on_export_completed)
            self.database_worker.error_occurred.connect(self.on_error_occurred)
            
            self.database_worker.set_operation("export", {
                "data": results,
                "format": format_type,
                "file_path": file_path
            })
            self.database_worker.start()
    
    def on_export_completed(self, step: str, data: Dict[str, Any]):
        """Handle export completion"""
        # Record successful export
        record_user_action('database_export_completed', 'database_tab',
                          data={
                              'file_path': data.get('file_path', ''),
                              'format': data.get('format', ''),
                              'export_successful': True
                          })
        
        file_path = data.get('file_path', '')
        format_type = data.get('format', '')
        
        self.status_label.setText(f"Export completed: {file_path}")
        QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")

    def load_statistics(self):
        """Load database statistics"""
        self.database_worker = DatabaseWorker()
        self.database_worker.statistics_updated.connect(self.on_statistics_updated)
        self.database_worker.error_occurred.connect(self.on_error_occurred)
        
        self.database_worker.set_operation("statistics", {})
        self.database_worker.start()
    
    def on_statistics_updated(self, stats: Dict[str, Any]):
        """Handle statistics update"""
        self.dashboard_widget.update_statistics(stats)
        logger.info("Database statistics updated")
    
    def refresh_database(self):
        """Refresh database connection and statistics"""
        # Record refresh action
        record_user_action('database_refreshed', 'database_tab',
                          data={
                              'timestamp': datetime.now().isoformat()
                          })
        
        self.status_label.setText("Refreshing database...")
        self.load_statistics()
        
        # Clear current results
        self.current_results.clear()
        self.results_widget.set_results([])
        
        # Reset stepper
        self.stepper.set_step(0)
        
        # Switch to dashboard
        self.content_tabs.setCurrentIndex(0)
    
    def closeEvent(self, event):
        """Handle close event"""
        # Cancel any running operations
        if self.database_worker and self.database_worker.isRunning():
            self.database_worker.cancel()
            self.database_worker.wait(3000)  # Wait up to 3 seconds
        
        event.accept()