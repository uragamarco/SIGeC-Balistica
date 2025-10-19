"""
StepperWidget - Modern step navigation component
Provides clear visual indication of current step and progress
"""

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                             QPushButton, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont
from typing import List, Optional


class StepIndicator(QWidget):
    """Individual step indicator with modern design"""
    
    def __init__(self, step_number: int, title: str, description: str = ""):
        super().__init__()
        self.step_number = step_number
        self.title = title
        self.description = description
        self.is_active = False
        self.is_completed = False
        self.is_enabled = True
        
        self.setFixedSize(200, 80)
        self.setProperty("class", "step-indicator")
        
    def set_active(self, active: bool):
        """Set step as active"""
        self.is_active = active
        self.update()
        
    def set_completed(self, completed: bool):
        """Set step as completed"""
        self.is_completed = completed
        self.update()
        
    def set_enabled(self, enabled: bool):
        """Set step as enabled/disabled"""
        self.is_enabled = enabled
        self.update()
        
    def paintEvent(self, event):
        """Custom paint event for modern step indicator"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Colors based on state
        if self.is_completed:
            circle_color = QColor("#10B981")  # Green
            text_color = QColor("#065F46")
            bg_color = QColor("#D1FAE5")
        elif self.is_active:
            circle_color = QColor("#3B82F6")  # Blue
            text_color = QColor("#1E40AF")
            bg_color = QColor("#DBEAFE")
        elif self.is_enabled:
            circle_color = QColor("#9CA3AF")  # Gray
            text_color = QColor("#374151")
            bg_color = QColor("#F9FAFB")
        else:
            circle_color = QColor("#D1D5DB")  # Light gray
            text_color = QColor("#9CA3AF")
            bg_color = QColor("#F9FAFB")
            
        # Draw background
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 8, 8)
        
        # Draw circle
        circle_rect = self.rect().adjusted(10, 10, -150, -50)
        painter.setBrush(QBrush(circle_color))
        painter.drawEllipse(circle_rect)
        
        # Draw step number or checkmark
        painter.setPen(QPen(QColor("white"), 2))
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        
        if self.is_completed:
            # Draw checkmark
            painter.drawText(circle_rect, Qt.AlignCenter, "✓")
        else:
            painter.drawText(circle_rect, Qt.AlignCenter, str(self.step_number))
            
        # Draw title
        title_rect = self.rect().adjusted(50, 10, -10, -40)
        painter.setPen(QPen(text_color, 1))
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignTop, self.title)
        
        # Draw description
        if self.description:
            desc_rect = self.rect().adjusted(50, 30, -10, -10)
            font = QFont("Arial", 8)
            painter.setFont(font)
            painter.setPen(QPen(text_color.lighter(120), 1))
            painter.drawText(desc_rect, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, 
                           self.description)


class StepperWidget(QWidget):
    """Modern stepper widget for step-by-step navigation"""
    
    stepChanged = pyqtSignal(int)  # Emitted when step changes
    stepActivated = pyqtSignal(str)  # Emitted when a step is activated (with step ID)
    
    def __init__(self, steps: List[dict], parent=None):
        """
        Initialize stepper widget
        
        Args:
            steps: List of step dictionaries with keys: 'title', 'description'
        """
        super().__init__(parent)
        self.steps = steps
        self.current_step = 0
        self.step_indicators = []
        # Compatibility alias for legacy code expecting 'step_widgets'
        self.step_widgets = self.step_indicators
        
        self.setup_ui()
        self.update_steps()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(0)
        
        # Steps container
        steps_container = QWidget()
        steps_layout = QHBoxLayout(steps_container)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(10)
        
        # Create step indicators
        for i, step in enumerate(self.steps):
            indicator = StepIndicator(
                i + 1, 
                step.get('title', f'Paso {i + 1}'),
                step.get('description', '')
            )
            self.step_indicators.append(indicator)
            steps_layout.addWidget(indicator)
            
            # Add connector line (except for last step)
            if i < len(self.steps) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setProperty("class", "step-connector")
                line.setFixedHeight(2)
                line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                steps_layout.addWidget(line)
                
        steps_layout.addStretch()
        layout.addWidget(steps_container)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 20, 0, 0)
        
        self.prev_button = QPushButton("← Anterior")
        self.prev_button.setProperty("class", "stepper-nav-button secondary")
        self.prev_button.clicked.connect(self.previous_step)
        self.prev_button.setEnabled(False)
        
        nav_layout.addWidget(self.prev_button)
        nav_layout.addStretch()
        
        self.next_button = QPushButton("Siguiente →")
        self.next_button.setProperty("class", "stepper-nav-button primary")
        self.next_button.clicked.connect(self.next_step)
        
        nav_layout.addWidget(self.next_button)
        layout.addLayout(nav_layout)
        
    def set_current_step(self, step):
        """Set the current active step; accepts index or step id/title."""
        # Map string id/title to index if needed
        if isinstance(step, str):
            idx = None
            for i, s in enumerate(self.steps):
                if isinstance(s, dict):
                    if s.get('id') == step or s.get('title') == step:
                        idx = i
                        break
            if idx is None:
                print(f"Warning: Unknown step '{step}'")
                return
            step = idx
        
        if 0 <= step < len(self.steps):
            self.current_step = step
            self.update_steps()
            self.stepChanged.emit(step)
            
            # Emit stepActivated signal with step ID if available
            if hasattr(self.steps[step], 'get') and 'id' in self.steps[step]:
                step_id = self.steps[step]['id']
            else:
                step_id = f"step_{step}"
            self.stepActivated.emit(step_id)
        
    def next_step(self):
        """Go to next step"""
        if self.current_step < len(self.steps) - 1:
            # Mark current step as completed
            self.step_indicators[self.current_step].set_completed(True)
            self.set_current_step(self.current_step + 1)
            
    def previous_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            # Mark current step as not completed
            self.step_indicators[self.current_step].set_completed(False)
            self.set_current_step(self.current_step - 1)
            
    def update_steps(self):
        """Update visual state of all steps"""
        for i, indicator in enumerate(self.step_indicators):
            if i < self.current_step:
                indicator.set_completed(True)
                indicator.set_active(False)
                indicator.set_enabled(True)
            elif i == self.current_step:
                indicator.set_completed(False)
                indicator.set_active(True)
                indicator.set_enabled(True)
            else:
                indicator.set_completed(False)
                indicator.set_active(False)
                indicator.set_enabled(False)
                
        # Update navigation buttons
        self.prev_button.setEnabled(self.current_step > 0)
        self.next_button.setEnabled(self.current_step < len(self.steps) - 1)
        
        if self.current_step == len(self.steps) - 1:
            self.next_button.setText("Finalizar")
        else:
            self.next_button.setText("Siguiente →")
            
    def get_current_step(self) -> int:
        """Get current step index"""
        return self.current_step
    
    def get_current_step_id(self) -> str:
        """Helper: retorna el id del paso actual sin cambiar contratos existentes."""
        idx = self.get_current_step()
        if 0 <= idx < len(self.steps):
            step = self.steps[idx]
            try:
                # steps are dicts; prefer explicit id
                step_id = step.get('id')
                if step_id:
                    return step_id
            except Exception:
                pass
        return f"step_{idx}"
    
    def is_last_step(self) -> bool:
        """Check if current step is the last one"""
        return self.current_step == len(self.steps) - 1
        
    def mark_step_completed(self, step: int):
        """Mark a specific step as completed"""
        if 0 <= step < len(self.step_indicators):
            self.step_indicators[step].set_completed(True)

    def set_step_completed(self, step):
        """Compatibility alias: mark a step completed by index or id/title."""
        if isinstance(step, str):
            for i, s in enumerate(self.steps):
                if isinstance(s, dict) and (s.get('id') == step or s.get('title') == step):
                    step = i
                    break
        if isinstance(step, int):
            self.mark_step_completed(step)

    def enable_step(self, step):
        """Enable a step by index or id/title for navigation/validation."""
        if isinstance(step, str):
            for i, s in enumerate(self.steps):
                if isinstance(s, dict) and (s.get('id') == step or s.get('title') == step):
                    step = i
                    break
        if isinstance(step, int) and 0 <= step < len(self.step_indicators):
            self.step_indicators[step].set_enabled(True)
            self.update_steps()

    def disable_step(self, step):
        """Disable a step by index or id/title for navigation/validation."""
        if isinstance(step, str):
            for i, s in enumerate(self.steps):
                if isinstance(s, dict) and (s.get('id') == step or s.get('title') == step):
                    step = i
                    break
        if isinstance(step, int) and 0 <= step < len(self.step_indicators):
            self.step_indicators[step].set_enabled(False)
            self.update_steps()