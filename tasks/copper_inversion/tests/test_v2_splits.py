import unittest
import numpy as np
from tasks.copper_inversion.experiments.v2.split_audit import check_records,shape_fingerprint


class SplitAudit(unittest.TestCase):
    def test_collisions(self):
        a=dict(split="train",parent_id="a",lineage_id="root-a",truth_sha256="truth-a",shape_fingerprint="shape-a")
        b=dict(split="test_id",parent_id="b",lineage_id="root-b",truth_sha256="truth-b",shape_fingerprint="shape-b")
        self.assertEqual(check_records([a,b]),{"train":1,"test_id":1})
        for key in ("parent_id","lineage_id","truth_sha256","shape_fingerprint"):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,key):
                check_records([a,{**b,key:a[key]}])

    def test_rotated_duplicate_screen(self):
        mask=np.zeros((9,8,7),bool);mask[2:6,1:4,3:6]=True;mask[4,4,4]=True
        transformed=np.transpose(mask,(2,0,1))[::-1]
        self.assertEqual(shape_fingerprint(mask),shape_fingerprint(transformed))
        other=mask.copy();other[1:3,5:7,1:3]=True
        self.assertNotEqual(shape_fingerprint(mask),shape_fingerprint(other))


if __name__=="__main__":unittest.main()
