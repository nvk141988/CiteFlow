import unittest
from cite_flow_core import normalize_text, extract_doi, extract_arxiv_id, calculate_similarity

class TestCrossrefUtils(unittest.TestCase):

    def test_normalize_text(self):
        self.assertEqual(normalize_text("Hello World!"), "hello world")
        # Code preserves spaces
        self.assertEqual(normalize_text("Title: Part 1"), "title part 1")

    def test_extract_doi(self):
        text = "Check this paper doi: 10.1000/12345/abc"
        self.assertEqual(extract_doi(text), "10.1000/12345/abc")
        
        text2 = "No DOI here"
        self.assertIsNone(extract_doi(text2))

    def test_extract_arxiv_id(self):
        text = "See arXiv:2312.01797 for details"
        self.assertEqual(extract_arxiv_id(text), "2312.01797")
        
        text2 = "Also as arXiv: 2101.12345v1"
        self.assertEqual(extract_arxiv_id(text2), "2101.12345")

    def test_calculate_similarity(self):
        s1 = "Deep Learning in Robotics"
        s2 = "Deep Learning in Robotics"
        self.assertEqual(calculate_similarity(s1, s2), 100)
        
        s3 = "Robotics with Deep Learning"
        # Token set ratio handles reordering well
        self.assertTrue(calculate_similarity(s1, s3) > 90)

if __name__ == '__main__':
    unittest.main()
