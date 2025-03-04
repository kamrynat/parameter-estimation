import unittest
import numpy as np
from SimplifiedThreePL import SimplifiedThreePL
from Experiment import Experiment

#Asked ChatGPT for help while generating code
class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        # Mock Experiment Data
        self.n_correct = np.array([30, 25, 20, 15, 10])
        self.n_incorrect = np.array([10, 15, 20, 25, 30])
        self.experiment = Experiment(self.n_correct, self.n_incorrect)
        self.model = SimplifiedThreePL(self.experiment)

    def test_initialization(self):
        self.assertIsInstance(self.model, SimplifiedThreePL)
        self.assertEqual(self.model.experiment, self.experiment)
        self.assertIsNone(self.model._discrimination)
        self.assertIsNone(self.model._logit_base_rate)
        self.assertFalse(self.model._is_fitted)
    
    def test_summary(self):
        summary = self.model.summary()
        self.assertEqual(summary["n_total"], sum(self.n_correct) + sum(self.n_incorrect))
        self.assertEqual(summary["n_correct"], sum(self.n_correct))
        self.assertEqual(summary["n_incorrect"], sum(self.n_incorrect))
        self.assertEqual(summary["n_conditions"], len(self.n_correct))
    
    def test_predict(self):
        parameters = [1.0, 0.0]  # Test parameters
        probabilities = self.model.predict(parameters)
        self.assertEqual(len(probabilities), len(self.n_correct))
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

