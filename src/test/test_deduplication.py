import unittest

from ..deduplication import *

class TestDeduplication(unittest.TestCase):

    def test_remove_partial_ngrams(self):
        # Do we want to treat partial overlap differently?
        lengths = [[1, 1, 2, 3], [1, 2, 3, 2, 2]]
        deduped = remove_partial_ngrams(lengths)
        self.assertEqual(deduped, [[1, 3], [3, 2, 2]])

if __name__ == "__main__":
    unittest.main()