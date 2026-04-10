import unittest
import pandas as pd
import numpy as np
from src.features import add_rolling_mean

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
    
    def test_add_rolling_mean(self):
        result = add_rolling_mean(self.df, 'col', 3)
        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
        pd.testing.assert_series_equal(result['col_rolling_mean_3'], expected, check_names=False)

if __name__ == '__main__':
    unittest.main()