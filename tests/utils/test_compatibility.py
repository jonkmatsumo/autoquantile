import unittest
import sys
from unittest.mock import patch, MagicMock
from src.utils.compatibility import apply_backward_compatibility

class TestCompatibility(unittest.TestCase):
    def setUp(self):
        # Clean up modules if they exist to test clean application
        self._remove_modules()

    def tearDown(self):
        self._remove_modules()

    def _remove_modules(self):
        for mod in ['src.model', 'src.model.model', 'src.model.preprocessing']:
            if mod in sys.modules:
                del sys.modules[mod]

    def test_apply_backward_compatibility(self):
        # Pre-condition: src.model doesn't exist
        self.assertNotIn('src.model', sys.modules)
        
        # Action
        apply_backward_compatibility()
        
        # Post-condition: src.model maps to src.xgboost
        self.assertIn('src.model', sys.modules)
        import src.xgboost
        self.assertEqual(sys.modules['src.model'], src.xgboost)
        
        # Verify submodules
        import src.xgboost.model
        self.assertEqual(sys.modules['src.model.model'], src.xgboost.model)

    def test_apply_backward_compatibility_idempotent(self):
        # It should handle being called twice safely
        apply_backward_compatibility()
        mod1 = sys.modules['src.model']
        
        apply_backward_compatibility()
        mod2 = sys.modules['src.model']
        
        self.assertEqual(mod1, mod2)
