"""
Test suite for set_params() hybrid parameter handling.

This module tests the set_params() method for both CalibratedForest and
CalibratedTree, ensuring proper handling of hybrid parameters from both
the calibrated models and the underlying sklearn estimators.

Note
----
This test suite can run with or without pytest. If pytest is not available,
it will run as a standalone script.
"""

import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from califorest import CalibratedForest, CalibratedTree


def get_sample_data():
    """Generate sample classification data for testing.
    
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    X, y = make_classification(
        n_samples=500, 
        n_features=10, 
        n_classes=2, 
        random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


class TestCalibratedForestSetParams:
    """Test suite for CalibratedForest.set_params()."""
    
    def test_basic_initialization(self):
        """Test basic initialization and get_params()."""
        clf = CalibratedForest(n_estimators=50, max_depth=5, random_state=42)
        params = clf.get_params()
        
        assert params['n_estimators'] == 50
        assert params['max_depth'] == 5
        assert params['random_state'] == 42
        print("  ✓ test_basic_initialization passed")
    
    def test_direct_parameters(self):
        """Test setting direct CalibratedForest parameters."""
        clf = CalibratedForest()
        clf.set_params(n_estimators=100, max_depth=10, ctype='logistic')
        params = clf.get_params()
        
        assert params['n_estimators'] == 100
        assert params['max_depth'] == 10
        assert params['ctype'] == 'logistic'
        print("  ✓ test_direct_parameters passed")
    
    def test_sklearn_parameters(self):
        """Test setting RandomForestClassifier parameters."""
        clf = CalibratedForest()
        clf.set_params(
            min_weight_fraction_leaf=0.1,
            max_features=0.8,
            n_jobs=-1,
            max_samples=0.9
        )
        params = clf.get_params()
        
        assert params['min_weight_fraction_leaf'] == 0.1
        assert params['max_features'] == 0.8
        assert params['n_jobs'] == -1
        assert params['max_samples'] == 0.9
        print("  ✓ test_sklearn_parameters passed")
    
    def test_hybrid_parameters(self):
        """Test setting both direct and sklearn parameters together."""
        params_to_set = {
            'n_estimators': 296,
            'max_depth': None,
            'ctype': 'isotonic',
            'test_size': 0.3,
            'random_state': 42,
            'min_weight_fraction_leaf': 0.0,
            'max_features': 1,
            'n_jobs': -1,
            'max_samples': 1.0
        }
        
        clf = CalibratedForest()
        clf.set_params(**params_to_set)
        result_params = clf.get_params()
        
        for key, value in params_to_set.items():
            assert result_params[key] == value
        print("  ✓ test_hybrid_parameters passed")
    
    def test_training_after_set_params(self):
        """Test that model can be trained after set_params()."""
        X_train, X_test, y_train, y_test = get_sample_data()
        
        clf = CalibratedForest()
        clf.set_params(
            n_estimators=50,
            max_depth=5,
            min_weight_fraction_leaf=0.0,
            max_features=0.8
        )
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)
        
        assert predictions.shape[0] == X_test.shape[0]
        assert probabilities.shape == (X_test.shape[0], 2)
        print("  ✓ test_training_after_set_params passed")
    
    def test_invalid_parameter(self):
        """Test that invalid parameters raise ValueError."""
        clf = CalibratedForest()
        
        try:
            clf.set_params(invalid_parameter=123)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Invalid parameter" in str(e)
        print("  ✓ test_invalid_parameter passed")
    
    def test_bayesian_optimization_scenario(self):
        """Test the exact scenario from BayesSearchCV."""
        best_params = {
            'n_estimators': 296,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'ctype': 'isotonic',
            'test_size': 0.3,
            'verbose': False,
            'random_state': 42,
            'min_weight_fraction_leaf': 0.0,
            'max_features': 1,
            'n_jobs': -1,
            'max_samples': 1.0
        }
        
        clf = CalibratedForest()
        clf = clf.set_params(**best_params)
        
        result_params = clf.get_params()
        for key, value in best_params.items():
            assert result_params[key] == value
        print("  ✓ test_bayesian_optimization_scenario passed")
    
    def run_all(self):
        """Run all tests in this class."""
        print("\nTestCalibratedForestSetParams:")
        self.test_basic_initialization()
        self.test_direct_parameters()
        self.test_sklearn_parameters()
        self.test_hybrid_parameters()
        self.test_training_after_set_params()
        self.test_invalid_parameter()
        self.test_bayesian_optimization_scenario()


class TestCalibratedTreeSetParams:
    """Test suite for CalibratedTree.set_params()."""
    
    def test_basic_initialization(self):
        """Test basic initialization and get_params()."""
        clf = CalibratedTree(n_estimators=100, max_depth=5, random_state=42)
        params = clf.get_params()
        
        assert params['n_estimators'] == 100
        assert params['max_depth'] == 5
        assert params['random_state'] == 42
        print("  ✓ test_basic_initialization passed")
    
    def test_direct_parameters(self):
        """Test setting direct CalibratedTree parameters."""
        clf = CalibratedTree()
        clf.set_params(
            n_estimators=200, 
            max_depth=10, 
            ctype='logistic', 
            alpha0=50
        )
        params = clf.get_params()
        
        assert params['n_estimators'] == 200
        assert params['max_depth'] == 10
        assert params['ctype'] == 'logistic'
        assert params['alpha0'] == 50
        print("  ✓ test_direct_parameters passed")
    
    def test_sklearn_parameters(self):
        """Test setting DecisionTreeClassifier parameters."""
        clf = CalibratedTree()
        clf.set_params(
            min_weight_fraction_leaf=0.1,
            max_leaf_nodes=50,
            ccp_alpha=0.01
        )
        params = clf.get_params()
        
        assert params['min_weight_fraction_leaf'] == 0.1
        assert params['max_leaf_nodes'] == 50
        assert params['ccp_alpha'] == 0.01
        print("  ✓ test_sklearn_parameters passed")
    
    def test_hybrid_parameters(self):
        """Test setting both direct and sklearn parameters together."""
        params_to_set = {
            'n_estimators': 150,
            'criterion': 'entropy',
            'max_depth': 7,
            'ctype': 'isotonic',
            'alpha0': 100,
            'beta0': 25,
            'random_state': 42,
            'min_weight_fraction_leaf': 0.1,
            'max_leaf_nodes': 50,
            'ccp_alpha': 0.01
        }
        
        clf = CalibratedTree()
        clf.set_params(**params_to_set)
        result_params = clf.get_params()
        
        for key, value in params_to_set.items():
            assert result_params[key] == value
        print("  ✓ test_hybrid_parameters passed")
    
    def test_training_after_set_params(self):
        """Test that model can be trained after set_params()."""
        X_train, X_test, y_train, y_test = get_sample_data()
        
        clf = CalibratedTree()
        clf.set_params(
            n_estimators=50,
            max_depth=5,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=50
        )
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)
        
        assert predictions.shape[0] == X_test.shape[0]
        assert probabilities.shape == (X_test.shape[0], 2)
        print("  ✓ test_training_after_set_params passed")
    
    def test_invalid_parameter(self):
        """Test that invalid parameters raise ValueError."""
        clf = CalibratedTree()
        
        try:
            clf.set_params(invalid_parameter=123)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Invalid parameter" in str(e)
        print("  ✓ test_invalid_parameter passed")
    
    def run_all(self):
        """Run all tests in this class."""
        print("\nTestCalibratedTreeSetParams:")
        self.test_basic_initialization()
        self.test_direct_parameters()
        self.test_sklearn_parameters()
        self.test_hybrid_parameters()
        self.test_training_after_set_params()
        self.test_invalid_parameter()


class TestParameterPersistence:
    """Test that parameters persist correctly through get/set cycles."""
    
    def test_forest_parameter_roundtrip(self):
        """Test CalibratedForest parameter roundtrip."""
        original_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'ctype': 'logistic',
            'min_weight_fraction_leaf': 0.05,
            'max_features': 0.7,
            'n_jobs': -1
        }
        
        clf1 = CalibratedForest()
        clf1.set_params(**original_params)
        
        # Get params and create new instance
        retrieved_params = clf1.get_params()
        clf2 = CalibratedForest()
        clf2.set_params(**retrieved_params)
        
        # Verify both have same parameters
        params1 = clf1.get_params()
        params2 = clf2.get_params()
        
        for key in original_params.keys():
            assert params1[key] == params2[key]
        print("  ✓ test_forest_parameter_roundtrip passed")
    
    def test_tree_parameter_roundtrip(self):
        """Test CalibratedTree parameter roundtrip."""
        original_params = {
            'n_estimators': 150,
            'criterion': 'entropy',
            'max_depth': 6,
            'ctype': 'isotonic',
            'min_weight_fraction_leaf': 0.1,
            'max_leaf_nodes': 40
        }
        
        clf1 = CalibratedTree()
        clf1.set_params(**original_params)
        
        # Get params and create new instance
        retrieved_params = clf1.get_params()
        clf2 = CalibratedTree()
        clf2.set_params(**retrieved_params)
        
        # Verify both have same parameters
        params1 = clf1.get_params()
        params2 = clf2.get_params()
        
        for key in original_params.keys():
            assert params1[key] == params2[key]
        print("  ✓ test_tree_parameter_roundtrip passed")
    
    def run_all(self):
        """Run all tests in this class."""
        print("\nTestParameterPersistence:")
        self.test_forest_parameter_roundtrip()
        self.test_tree_parameter_roundtrip()


def run_all_tests():
    """Run all test suites."""
    print("=" * 80)
    print("Running set_params() Test Suite")
    print("=" * 80)
    
    try:
        # Run CalibratedForest tests
        forest_tests = TestCalibratedForestSetParams()
        forest_tests.run_all()
        
        # Run CalibratedTree tests
        tree_tests = TestCalibratedTreeSetParams()
        tree_tests.run_all()
        
        # Run parameter persistence tests
        persistence_tests = TestParameterPersistence()
        persistence_tests.run_all()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Try to use pytest if available, otherwise run standalone
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        sys.exit(run_all_tests())