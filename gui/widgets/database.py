"""
Database-specific widgets for the ballistic database management interface
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QScrollArea, QSplitter, QFrame, QSpacerItem,
    QSizePolicy, QTabWidget, QListWidget, QListWidgetItem, QSlider,
    QDoubleSpinBox, QDateEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QTreeWidget, QTreeWidgetItem, QButtonGroup,
    QRadioButton, QCalendarWidget, QProgressBar, QMenu, QAction
)
from PyQt5.QtCore import Qt, pyqtSignal, QDate, QSize, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPainter, QPen, QColor, QBrush
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class DatabaseStepper(QWidget):
    """
    Specialized stepper for database workflow navigation
    """
    
    stepChanged = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_step = 0
        self.steps = [
            {"title": "Search", "description": "Configure search parameters"},
            {"title": "Filter", "description": "Apply advanced filters"},
            {"title": "Results", "description": "View and analyze results"},
            {"title": "Export", "description": "Export or save results"}
        ]
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Database Workflow")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Steps container
        steps_container = QWidget()
        steps_layout = QVBoxLayout(steps_container)
        steps_layout.setSpacing(5)
        
        self.step_widgets = []
        for i, step in enumerate(self.steps):
            step_widget = self.create_step_widget(i, step)
            self.step_widgets.append(step_widget)
            steps_layout.addWidget(step_widget)
        
        layout.addWidget(steps_container)
        layout.addStretch()
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self.previous_step)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.next_step)
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        
        self.update_step_display()
    
    def create_step_widget(self, index: int, step: Dict[str, str]) -> QWidget:
        widget = QFrame()
        widget.setObjectName(f"step_{index}")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Step number
        number_label = QLabel(str(index + 1))
        number_label.setFixedSize(24, 24)
        number_label.setAlignment(Qt.AlignCenter)
        number_label.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                border-radius: 12px;
                font-weight: bold;
                color: #7f8c8d;
            }
        """)
        layout.addWidget(number_label)
        
        # Step content
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)
        
        title_label = QLabel(step["title"])
        title_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        content_layout.addWidget(title_label)
        
        desc_label = QLabel(step["description"])
        desc_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        content_layout.addWidget(desc_label)
        
        layout.addLayout(content_layout)
        layout.addStretch()
        
        return widget
    
    def update_step_display(self):
        for i, widget in enumerate(self.step_widgets):
            number_label = widget.findChild(QLabel)
            if i == self.current_step:
                widget.setStyleSheet("QFrame { background-color: #e8f4fd; border-left: 3px solid #3498db; }")
                number_label.setStyleSheet("""
                    QLabel {
                        background-color: #3498db;
                        border-radius: 12px;
                        font-weight: bold;
                        color: white;
                    }
                """)
            elif i < self.current_step:
                widget.setStyleSheet("QFrame { background-color: #d5f4e6; border-left: 3px solid #27ae60; }")
                number_label.setStyleSheet("""
                    QLabel {
                        background-color: #27ae60;
                        border-radius: 12px;
                        font-weight: bold;
                        color: white;
                    }
                """)
            else:
                widget.setStyleSheet("QFrame { background-color: #f8f9fa; border-left: 3px solid #ecf0f1; }")
                number_label.setStyleSheet("""
                    QLabel {
                        background-color: #ecf0f1;
                        border-radius: 12px;
                        font-weight: bold;
                        color: #7f8c8d;
                    }
                """)
        
        # Update navigation buttons
        self.prev_btn.setEnabled(self.current_step > 0)
        self.next_btn.setEnabled(self.current_step < len(self.steps) - 1)
    
    def next_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_step_display()
            self.stepChanged.emit(self.current_step)
    
    def previous_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_step_display()
            self.stepChanged.emit(self.current_step)
    
    def set_step(self, step: int):
        if 0 <= step < len(self.steps):
            self.current_step = step
            self.update_step_display()
            self.stepChanged.emit(self.current_step)


class DatabaseSearchWidget(QWidget):
    """
    Advanced search widget for ballistic database
    """
    
    searchRequested = pyqtSignal(dict)
    filtersChanged = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.search_filters = {}
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Search input
        search_group = QGroupBox("Text Search")
        search_layout = QVBoxLayout(search_group)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in all fields...")
        self.search_input.textChanged.connect(self.on_search_changed)
        search_layout.addWidget(self.search_input)
        
        # Search options
        options_layout = QHBoxLayout()
        self.case_sensitive_cb = QCheckBox("Case sensitive")
        self.whole_words_cb = QCheckBox("Whole words")
        self.regex_cb = QCheckBox("Regex")
        
        options_layout.addWidget(self.case_sensitive_cb)
        options_layout.addWidget(self.whole_words_cb)
        options_layout.addWidget(self.regex_cb)
        options_layout.addStretch()
        
        search_layout.addLayout(options_layout)
        layout.addWidget(search_group)
        
        # Ballistic filters
        ballistic_group = QGroupBox("Ballistic Filters")
        ballistic_layout = QGridLayout(ballistic_group)
        
        # Caliber filter
        ballistic_layout.addWidget(QLabel("Caliber:"), 0, 0)
        self.caliber_combo = QComboBox()
        self.caliber_combo.addItems(["All", ".22", ".38", ".45", "9mm", "Other"])
        self.caliber_combo.currentTextChanged.connect(self.on_filters_changed)
        ballistic_layout.addWidget(self.caliber_combo, 0, 1)
        
        # Weapon type filter
        ballistic_layout.addWidget(QLabel("Weapon Type:"), 1, 0)
        self.weapon_combo = QComboBox()
        self.weapon_combo.addItems(["All", "Pistol", "Rifle", "Revolver", "Other"])
        self.weapon_combo.currentTextChanged.connect(self.on_filters_changed)
        ballistic_layout.addWidget(self.weapon_combo, 1, 1)
        
        # Manufacturer filter
        ballistic_layout.addWidget(QLabel("Manufacturer:"), 2, 0)
        self.manufacturer_combo = QComboBox()
        self.manufacturer_combo.addItems(["All", "Glock", "Smith & Wesson", "Colt", "Other"])
        self.manufacturer_combo.currentTextChanged.connect(self.on_filters_changed)
        ballistic_layout.addWidget(self.manufacturer_combo, 2, 1)
        
        layout.addWidget(ballistic_group)
        
        # Date range filter
        date_group = QGroupBox("Date Range")
        date_layout = QGridLayout(date_group)
        
        date_layout.addWidget(QLabel("From:"), 0, 0)
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addYears(-1))
        self.date_from.dateChanged.connect(self.on_filters_changed)
        date_layout.addWidget(self.date_from, 0, 1)
        
        date_layout.addWidget(QLabel("To:"), 1, 0)
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.dateChanged.connect(self.on_filters_changed)
        date_layout.addWidget(self.date_to, 1, 1)
        
        layout.addWidget(date_group)
        
        # Search button
        self.search_btn = QPushButton("🔍 Search Database")
        self.search_btn.clicked.connect(self.perform_search)
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        layout.addWidget(self.search_btn)
        
        # Clear filters button
        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self.clear_filters)
        layout.addWidget(clear_btn)
        
        layout.addStretch()
    
    def on_search_changed(self):
        self.search_filters['text'] = self.search_input.text()
        self.filtersChanged.emit(self.search_filters)
    
    def on_filters_changed(self):
        self.search_filters.update({
            'caliber': self.caliber_combo.currentText(),
            'weapon_type': self.weapon_combo.currentText(),
            'manufacturer': self.manufacturer_combo.currentText(),
            'date_from': self.date_from.date().toString(Qt.ISODate),
            'date_to': self.date_to.date().toString(Qt.ISODate),
            'case_sensitive': self.case_sensitive_cb.isChecked(),
            'whole_words': self.whole_words_cb.isChecked(),
            'regex': self.regex_cb.isChecked()
        })
        self.filtersChanged.emit(self.search_filters)
    
    def perform_search(self):
        self.on_filters_changed()
        self.searchRequested.emit(self.search_filters)
    
    def clear_filters(self):
        self.search_input.clear()
        self.caliber_combo.setCurrentIndex(0)
        self.weapon_combo.setCurrentIndex(0)
        self.manufacturer_combo.setCurrentIndex(0)
        self.date_from.setDate(QDate.currentDate().addYears(-1))
        self.date_to.setDate(QDate.currentDate())
        self.case_sensitive_cb.setChecked(False)
        self.whole_words_cb.setChecked(False)
        self.regex_cb.setChecked(False)
        self.search_filters.clear()
        self.filtersChanged.emit(self.search_filters)


class DatabaseResultsWidget(QWidget):
    """
    Results display widget for database search
    """
    
    itemSelected = pyqtSignal(dict)
    itemDoubleClicked = pyqtSignal(dict)
    exportRequested = pyqtSignal(list, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results = []
        self.selected_items = []
        self.view_mode = "grid"
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        self.results_label = QLabel("No results")
        self.results_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(self.results_label)
        
        header_layout.addStretch()
        
        # View mode buttons
        view_group = QButtonGroup(self)
        
        self.grid_btn = QPushButton("Grid")
        self.grid_btn.setCheckable(True)
        self.grid_btn.setChecked(True)
        self.grid_btn.clicked.connect(lambda: self.set_view_mode("grid"))
        view_group.addButton(self.grid_btn)
        header_layout.addWidget(self.grid_btn)
        
        self.list_btn = QPushButton("List")
        self.list_btn.setCheckable(True)
        self.list_btn.clicked.connect(lambda: self.set_view_mode("list"))
        view_group.addButton(self.list_btn)
        header_layout.addWidget(self.list_btn)
        
        # Sort options
        header_layout.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Date", "Caliber", "Relevance", "Case ID"])
        self.sort_combo.currentTextChanged.connect(self.sort_results)
        header_layout.addWidget(self.sort_combo)
        
        layout.addLayout(header_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results area
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.results_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        
        # No results message
        self.no_results_label = QLabel("Perform a search to see results")
        self.no_results_label.setAlignment(Qt.AlignCenter)
        self.no_results_label.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 40px;")
        self.results_layout.addWidget(self.no_results_label)
        
        self.results_scroll.setWidget(self.results_widget)
        layout.addWidget(self.results_scroll)
        
        # Export controls
        export_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("📤 Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        
        self.export_format = QComboBox()
        self.export_format.addItems(["JSON", "CSV", "PDF", "Excel"])
        export_layout.addWidget(self.export_format)
        
        export_layout.addStretch()
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all)
        self.select_all_btn.setEnabled(False)
        export_layout.addWidget(self.select_all_btn)
        
        layout.addLayout(export_layout)
    
    def set_results(self, results: List[Dict]):
        self.results = results
        self.display_results()
        
        # Update UI state
        count = len(results)
        self.results_label.setText(f"{count} result{'s' if count != 1 else ''}")
        self.export_btn.setEnabled(count > 0)
        self.select_all_btn.setEnabled(count > 0)
    
    def display_results(self):
        # Clear existing results
        for i in reversed(range(self.results_layout.count())):
            child = self.results_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        if not self.results:
            self.no_results_label = QLabel("No results found")
            self.no_results_label.setAlignment(Qt.AlignCenter)
            self.no_results_label.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 40px;")
            self.results_layout.addWidget(self.no_results_label)
            return
        
        if self.view_mode == "grid":
            self.display_grid_results()
        else:
            self.display_list_results()
    
    def display_grid_results(self):
        # Create grid layout for results
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(10)
        
        cols = 3  # Number of columns
        for i, result in enumerate(self.results):
            row = i // cols
            col = i % cols
            
            result_card = self.create_result_card(result)
            grid_layout.addWidget(result_card, row, col)
        
        self.results_layout.addWidget(grid_widget)
        self.results_layout.addStretch()
    
    def display_list_results(self):
        # Create list layout for results
        for result in self.results:
            result_item = self.create_result_list_item(result)
            self.results_layout.addWidget(result_item)
        
        self.results_layout.addStretch()
    
    def create_result_card(self, result: Dict) -> QWidget:
        card = QFrame()
        card.setObjectName("resultCard")
        card.setStyleSheet("""
            QFrame#resultCard {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
            }
            QFrame#resultCard:hover {
                border-color: #3498db;
                box-shadow: 0 2px 8px rgba(52, 152, 219, 0.2);
            }
        """)
        card.setFixedSize(250, 200)
        card.mousePressEvent = lambda event: self.on_item_selected(result)
        card.mouseDoubleClickEvent = lambda event: self.itemDoubleClicked.emit(result)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Case ID
        case_id = QLabel(f"Case: {result.get('case_id', 'Unknown')}")
        case_id.setStyleSheet("font-weight: bold; color: #2c3e50;")
        layout.addWidget(case_id)
        
        # Caliber and weapon type
        details = QLabel(f"{result.get('caliber', 'Unknown')} • {result.get('weapon_type', 'Unknown')}")
        details.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(details)
        
        # Date
        date = QLabel(f"Date: {result.get('date', 'Unknown')}")
        date.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        layout.addWidget(date)
        
        # Thumbnail placeholder
        thumbnail = QLabel("📷 Image")
        thumbnail.setAlignment(Qt.AlignCenter)
        thumbnail.setStyleSheet("background-color: #f8f9fa; border: 1px dashed #ddd; padding: 20px;")
        layout.addWidget(thumbnail)
        
        layout.addStretch()
        
        return card
    
    def create_result_list_item(self, result: Dict) -> QWidget:
        item = QFrame()
        item.setObjectName("resultListItem")
        item.setStyleSheet("""
            QFrame#resultListItem {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 5px;
            }
            QFrame#resultListItem:hover {
                border-color: #3498db;
                background-color: #f8f9fa;
            }
        """)
        item.mousePressEvent = lambda event: self.on_item_selected(result)
        item.mouseDoubleClickEvent = lambda event: self.itemDoubleClicked.emit(result)
        
        layout = QHBoxLayout(item)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Main info
        info_layout = QVBoxLayout()
        
        case_id = QLabel(f"Case: {result.get('case_id', 'Unknown')}")
        case_id.setStyleSheet("font-weight: bold; color: #2c3e50;")
        info_layout.addWidget(case_id)
        
        details = QLabel(f"{result.get('caliber', 'Unknown')} • {result.get('weapon_type', 'Unknown')} • {result.get('manufacturer', 'Unknown')}")
        details.setStyleSheet("color: #7f8c8d;")
        info_layout.addWidget(details)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Date
        date = QLabel(result.get('date', 'Unknown'))
        date.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(date)
        
        return item
    
    def on_item_selected(self, result: Dict):
        self.itemSelected.emit(result)
    
    def set_view_mode(self, mode: str):
        self.view_mode = mode
        self.display_results()
    
    def sort_results(self):
        sort_key = self.sort_combo.currentText().lower()
        if sort_key in ['date', 'caliber', 'case_id']:
            self.results.sort(key=lambda x: x.get(sort_key.replace(' ', '_'), ''))
            self.display_results()
    
    def select_all(self):
        self.selected_items = self.results.copy()
        # Update visual selection state here
    
    def export_results(self):
        if self.results:
            format_type = self.export_format.currentText()
            self.exportRequested.emit(self.results, format_type)
    
    def show_loading(self, show: bool):
        self.progress_bar.setVisible(show)
        if show:
            self.progress_bar.setRange(0, 0)  # Indeterminate progress


class DatabaseDashboardWidget(QWidget):
    """
    Dashboard widget for database statistics and overview
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Database Dashboard")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;")
        layout.addWidget(title)
        
        # Statistics cards
        stats_layout = QGridLayout()
        
        # Total cases card
        total_card = self.create_stat_card("Total Cases", "1,234", "#3498db")
        stats_layout.addWidget(total_card, 0, 0)
        
        # Recent cases card
        recent_card = self.create_stat_card("Recent Cases", "45", "#27ae60")
        stats_layout.addWidget(recent_card, 0, 1)
        
        # Pending analysis card
        pending_card = self.create_stat_card("Pending Analysis", "12", "#f39c12")
        stats_layout.addWidget(pending_card, 0, 2)
        
        # Matches found card
        matches_card = self.create_stat_card("Matches Found", "89", "#e74c3c")
        stats_layout.addWidget(matches_card, 0, 3)
        
        layout.addLayout(stats_layout)
        
        # Charts area
        charts_widget = QTabWidget()
        
        # Caliber distribution chart
        caliber_tab = QWidget()
        caliber_layout = QVBoxLayout(caliber_tab)
        caliber_layout.addWidget(QLabel("Caliber Distribution Chart Placeholder"))
        charts_widget.addTab(caliber_tab, "📊 Calibers")
        
        # Timeline chart
        timeline_tab = QWidget()
        timeline_layout = QVBoxLayout(timeline_tab)
        timeline_layout.addWidget(QLabel("Timeline Chart Placeholder"))
        charts_widget.addTab(timeline_tab, "📅 Timeline")
        
        # Geographic distribution
        geo_tab = QWidget()
        geo_layout = QVBoxLayout(geo_tab)
        geo_layout.addWidget(QLabel("Geographic Distribution Placeholder"))
        charts_widget.addTab(geo_tab, "🗺️ Geographic")
        
        layout.addWidget(charts_widget)
    
    def create_stat_card(self, title: str, value: str, color: str) -> QWidget:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        
        value_label = QLabel(value)
        value_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        layout.addWidget(value_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        layout.addWidget(title_label)
        
        return card
    
    def update_statistics(self, stats: Dict[str, Any]):
        """Update dashboard with new statistics"""
        # Implementation would update the stat cards with real data
        pass


class DatabaseProgressWidget(QWidget):
    """
    Progress widget for database operations
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        self.title_label = QLabel("Database Operation")
        self.title_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        layout.addWidget(self.title_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        # Details area
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(100)
        self.details_text.setReadOnly(True)
        layout.addWidget(self.details_text)
    
    def set_progress(self, value: int, maximum: int = 100):
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
    
    def set_status(self, status: str):
        self.status_label.setText(status)
    
    def add_detail(self, detail: str):
        self.details_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {detail}")
    
    def clear_details(self):
        self.details_text.clear()
    
    def set_title(self, title: str):
        self.title_label.setText(title)