from pathlib import Path
import tempfile
import unittest
from tasks.antibody.tools.complex_screening.construct_controls import crop_a3m


class CroppingTests(unittest.TestCase):
    def test_insertions_do_not_consume_query_columns_and_empty_rows_drop(self):
        with tempfile.TemporaryDirectory() as folder:
            r=Path(folder);source=r/'input.a3m';out=r/'crop.a3m'
            source.write_text('>query\nACDEFG\n>insertions\nACxyD--G\n>no_overlap\n---EFG\n')
            meta=crop_a3m(source,'ACDEFG',3,out)
            self.assertEqual(out.read_text(),'>query\nACD\n>insertions\nACxyD\n')
            self.assertEqual(meta['rows_after'],2)
            source.write_text('>query\nACDEFG\n>bad_columns\nACD--\n')
            with self.assertRaisesRegex(ValueError,'columns'):crop_a3m(source,'ACDEFG',3,r/'bad.a3m')
            with self.assertRaisesRegex(ValueError,'query'):crop_a3m(source,'AAAAAA',3,r/'bad-query.a3m')


if __name__=='__main__':unittest.main()
