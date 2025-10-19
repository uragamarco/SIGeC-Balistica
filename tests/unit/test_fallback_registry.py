#!/usr/bin/env python3
"""
Tests unitarios para core/fallback_registry
"""

import unittest
import sys
from pathlib import Path

# Asegurar que el proyecto raíz esté en PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from core.fallback_registry import (
        FallbackRegistry,
        get_fallback,
        create_robust_import,
    )
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: core.fallback_registry import failed: {e}")
    CORE_AVAILABLE = False


@unittest.skipUnless(CORE_AVAILABLE, "Core fallback registry not available")
class TestCoreFallbackRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = FallbackRegistry()

    def test_registry_has_core_categories(self):
        available = self.registry.list_available_fallbacks()
        self.assertIsInstance(available, list)
        for category in [
            'deep_learning', 'web_service', 'image_processing', 'database',
            'torch', 'tensorflow', 'flask', 'rawpy', 'core_components'
        ]:
            self.assertIn(category, available)

    def test_get_fallback_function(self):
        dl = get_fallback('deep_learning')
        self.assertIsNotNone(dl)

    def test_register_custom_fallback(self):
        class DummyFallback:
            pass
        dummy = DummyFallback()
        self.registry.register_fallback('dummy_cat', dummy)
        self.assertIs(self.registry.get_fallback('dummy_cat'), dummy)

    def test_create_robust_import_success(self):
        importer = create_robust_import('json')
        mod = importer()
        import json as json_mod
        self.assertIs(mod, json_mod)

    def test_create_robust_import_with_fallback(self):
        importer = create_robust_import('package_that_does_not_exist_xyz', 'deep_learning')
        obj = importer()
        self.assertIsNotNone(obj)

    def test_create_robust_import_without_fallback(self):
        importer = create_robust_import('package_that_does_not_exist_xyz')
        with self.assertRaises(ImportError):
            importer()


if __name__ == '__main__':
    unittest.main(verbosity=2)