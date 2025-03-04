import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment
from src.SignalDetection import SignalDetection

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        # Mock SignalDetection Data
        self.hits = 30
        self.misses = 10
        self.false_alarms = 15
        self.correct_rejections = 25

        # Create SignalDetection object with correct parameters
        self.signal_detection = SignalDetection(self.hits, self.misses, self.false_alarms, self.correct_rejections)

        # Initialize Experiment and add SignalDetection object
        self.experiment = Experiment()
        self.experiment.add_condition(self.signal_detection)

        # Initialize SimplifiedThreePL with the corrected Experiment object
        self.model = SimplifiedThreePL(self.experiment)

    def test_initialization(self):
        self.assertIsInstance(self.model, SimplifiedThreePL)
        self.assertEqual(self.model.experiment, self.experiment)
        self.assertIsNone(self.model._discrimination)
        self.assertIsNone(self.model._logit_base_rate)
        self.assertFalse(self.model._is_fitted)
    
    def test_summary(self):
        summary = self.model.summary()
        total_correct = self.hits + self.correct_rejections
	total_incorrect = self.misses + self.false_alarms

	self.assertEqual(summary["n_total"], total_correct + total_incorrect)
	self.assertEqual(summary["n_correct"], total_correct)
	self.assertEqual(summary["n_incorrect"], total_incorrect)
	self.assertEqual(summary["n_conditions"], 1)  # Only one condition is added

    
    def test_predict(self):
        parameters = [1.0, 0.0]  # Test parameters
        probabilities = self.model.predict(parameters)
        self.assertEqual(len(probabilities), 5)  # Since difficulties array has 5 elements
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
    
    def test_negative_log_likelihood(self):
        parameters = [1.0, 0.0]
        nll = self.model.negative_log_likelihood(parameters)
        self.assertIsInstance(nll, float)
        self.assertGreater(nll, 0)
    
    def test_fit(self):
        self.model.fit()
        self.assertTrue(self.model._is_fitted)
        self.assertIsInstance(self.model._discrimination, float)
        self.assertIsInstance(self.model._logit_base_rate, float)
    
    def test_get_discrimination(self):
        self.model.fit()
        discrimination = self.model.get_discrimination()
        self.assertIsInstance(discrimination, float)
    
    def test_get_base_rate(self):
        self.model.fit()
        base_rate = self.model.get_base_rate()
        self.assertIsInstance(base_rate, float)
        self.assertTrue(0 <= base_rate <= 1)
    
    def test_get_discrimination_before_fit(self):
        with self.assertRaises(ValueError):
            self.model.get_discrimination()
    
    def test_get_base_rate_before_fit(self):
        with self.assertRaises(ValueError):
            self.model.get_base_rate()
    
if __name__ == "__main__":
    unittest.main()

