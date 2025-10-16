#!/usr/bin/env python3
"""
Simple GUI test to verify the main window functionality
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_gui():
    """Test basic GUI functionality"""
    
    print("Testing GUI functionality...")
    
    try:
        # Import PyQt5
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        
        print("✓ PyQt5 imported successfully")
        
        # Create application
        app = QApplication(sys.argv)
        print("✓ QApplication created")
        
        # Configure high DPI
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        
        if hasattr(Qt, 'AA_ShareOpenGLContexts'):
            QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
        
        print("✓ Qt attributes configured")
        
        # Import and create main window
        from gui.main_window import MainWindow
        print("✓ MainWindow imported")
        
        # Apply theme
        from gui.styles import apply_SIGeC_Balistica_theme
        apply_SIGeC_Balistica_theme(app)
        print("✓ Theme applied")
        
        # Create window
        window = MainWindow()
        print("✓ MainWindow created")
        
        # Show window
        window.show()
        print("✓ Window shown")
        
        # Center window
        screen = app.primaryScreen().geometry()
        window_size = window.geometry()
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2
        window.move(x, y)
        print("✓ Window centered")
        
        print("\n🎉 GUI test successful! Window should be visible.")
        print("Close the window to exit.")
        
        # Run event loop
        return app.exec_()
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = test_gui()
    sys.exit(exit_code)