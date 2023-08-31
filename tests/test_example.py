"""
Author:        David Meijer
Licence:       MIT License
Description:   Example test case.
Dependencies:  python>=3.10
"""
import unittest 

class TestExample(unittest.TestCase): 
    """
    Example test case.
    """
    def test_example(self) -> None: 
        """
        Example test.
        """
        self.assertEqual(1, 1)