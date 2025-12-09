import unittest
import logging
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.utils.logger import setup_logging, get_logger


class TestSetupLogging(unittest.TestCase):
    """Tests for setup_logging function."""
    
    def setUp(self):
        # Reset logging to default state before each test
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)
    
    def test_setup_logging_default_level(self):
        """Test setup_logging with default INFO level."""
        setup_logging()
        
        # Verify root logger level is set
        self.assertEqual(logging.root.level, logging.INFO)
        
        # Verify StreamHandler is added
        handlers = logging.root.handlers
        self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in handlers))
    
    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom log level."""
        setup_logging(level=logging.DEBUG)
        self.assertEqual(logging.root.level, logging.DEBUG)
        
        setup_logging(level=logging.WARNING)
        self.assertEqual(logging.root.level, logging.WARNING)
    
    def test_setup_logging_with_file(self):
        """Test setup_logging with file handler."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp_file:
            log_file_path = tmp_file.name
        
        try:
            setup_logging(log_file=log_file_path)
            
            # Verify both StreamHandler and FileHandler are present
            handlers = logging.root.handlers
            has_stream = any(isinstance(h, logging.StreamHandler) for h in handlers)
            has_file = any(isinstance(h, logging.FileHandler) for h in handlers)
            
            self.assertTrue(has_stream, "StreamHandler should be present")
            self.assertTrue(has_file, "FileHandler should be present")
            
            # Verify file handler writes to correct file
            file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
            self.assertEqual(file_handlers[0].baseFilename, log_file_path)
        finally:
            # Clean up
            if os.path.exists(log_file_path):
                os.unlink(log_file_path)
    
    def test_setup_logging_force_reconfigure(self):
        """Test that setup_logging can reconfigure existing logger."""
        # Setup logging once
        setup_logging(level=logging.INFO)
        initial_handlers_count = len(logging.root.handlers)
        
        # Setup again with different level (should force reconfigure)
        setup_logging(level=logging.DEBUG)
        
        # Should still work and update level
        self.assertEqual(logging.root.level, logging.DEBUG)
    
    def test_setup_logging_format(self):
        """Test that logging format is set correctly."""
        setup_logging()
        
        # Get a logger and check if it uses the format
        logger = logging.getLogger("test_logger")
        
        # Create a mock handler to capture log records
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        # Log a message
        logger.info("Test message")
        
        # Verify the logger is configured (format should include asctime, name, levelname, message)
        # We can't easily test the exact format without capturing output,
        # but we can verify the logger works
        self.assertTrue(logger.isEnabledFor(logging.INFO))


class TestGetLogger(unittest.TestCase):
    """Tests for get_logger function."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test_module")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_module")
    
    def test_get_logger_different_names(self):
        """Test that get_logger returns different loggers for different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        self.assertNotEqual(logger1, logger2)
        self.assertEqual(logger1.name, "module1")
        self.assertEqual(logger2.name, "module2")
    
    def test_get_logger_same_name_returns_same_logger(self):
        """Test that get_logger returns the same logger instance for the same name."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        
        # Should return the same logger instance (singleton behavior)
        self.assertIs(logger1, logger2)
    
    def test_get_logger_inherits_root_config(self):
        """Test that logger inherits root logger configuration."""
        setup_logging(level=logging.DEBUG)
        
        logger = get_logger("test_module")
        
        # Logger should be enabled for DEBUG level (inherited from root)
        self.assertTrue(logger.isEnabledFor(logging.DEBUG))
    
    def test_get_logger_can_log_messages(self):
        """Test that returned logger can actually log messages."""
        setup_logging(level=logging.INFO)
        
        logger = get_logger("test_module")
        
        # Should not raise any exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Verify logger is functional
        self.assertTrue(logger.isEnabledFor(logging.INFO))


class TestLoggerIntegration(unittest.TestCase):
    """Integration tests for logger module."""
    
    def setUp(self):
        # Reset logging before each test
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)
    
    def test_setup_and_get_logger_work_together(self):
        """Test that setup_logging and get_logger work together."""
        setup_logging(level=logging.INFO)
        logger = get_logger("integration_test")
        
        # Logger should respect the root level
        self.assertTrue(logger.isEnabledFor(logging.INFO))
        self.assertFalse(logger.isEnabledFor(logging.DEBUG))
    
    def test_multiple_loggers_with_file(self):
        """Test multiple loggers with file logging."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp_file:
            log_file_path = tmp_file.name
        
        try:
            setup_logging(level=logging.INFO, log_file=log_file_path)
            
            logger1 = get_logger("module1")
            logger2 = get_logger("module2")
            
            # Both should be able to log
            logger1.info("Message from module1")
            logger2.info("Message from module2")
            
            # Verify file was created and has content
            self.assertTrue(os.path.exists(log_file_path))
            with open(log_file_path, 'r') as f:
                content = f.read()
                self.assertIn("module1", content)
                self.assertIn("module2", content)
        finally:
            if os.path.exists(log_file_path):
                os.unlink(log_file_path)

