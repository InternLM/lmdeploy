import tempfile
import unittest
from pathlib import Path

from doc_link_checker import analyze_doc


class TestDocLinkChecker(unittest.TestCase):

    def test_existing_cross_file_target_with_anchor_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'target.md').write_text('# Section\n')
            source = root / 'source.md'
            source.write_text('[target](target.md#section)\n')

            analyze_doc(str(root), str(source))

    def test_missing_cross_file_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / 'source.md'
            source.write_text('[target](missing.md#section)\n')

            with self.assertRaisesRegex(Exception, 'found link error'):
                analyze_doc(str(root), str(source))


if __name__ == '__main__':
    unittest.main()
