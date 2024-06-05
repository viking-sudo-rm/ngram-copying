import unittest

from ..ngram_novelty import NgramNovelty

class TestNgramNovelty(unittest.TestCase):

    def test_get_proportion_unique(self):
        docs = [[1, 2, 3, 2]]
        nov = NgramNovelty(max_n=4)
        sizes, prop_unique = nov.get_proportion_unique(docs)
        self.assertEqual(sizes.tolist(), [1, 2, 3, 4])
        self.assertEqual(prop_unique.tolist(), [0 / 4, 0 / 3, 1 / 2, 1 / 1])

    def test_get_proportion_unique_multidoc(self):
        docs = [[1, 2, 2], [1, 2, 3, 3, 1]]
        nov = NgramNovelty(max_n=4)
        sizes, prop_unique = nov.get_proportion_unique(docs)
        self.assertEqual(sizes.tolist(), [1, 2, 3, 4])
        self.assertEqual(prop_unique.tolist(), [0 / 8, 1 / 6, 2 / 4, 2 / 2])

if __name__ == "__main__":
    unittest.main()