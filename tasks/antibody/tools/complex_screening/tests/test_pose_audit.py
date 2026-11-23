import importlib.util
from pathlib import Path
import tempfile
import unittest
from tasks.antibody.tools.complex_screening.pose_audit import audit


@unittest.skipUnless(importlib.util.find_spec('Bio') and importlib.util.find_spec('numpy'),'Predictor dependencies required')
class PoseAuditTests(unittest.TestCase):
    def test_rigid_motion_invariance_wrong_pose_and_wrong_crystal_copy(self):
        import numpy as np
        from Bio.PDB import Atom,Chain,Model,Residue,Structure,MMCIFIO
        chains={'A':(['ALA','CYS'],[[1,1,2],[2,2,2]]),'B':(['ASP','GLU'],[[-1,1,2],[-2,2,2]]),'Y':(['PHE','GLY','HIS'],[[0,0,0],[4,0,0],[0,4,0]]),'H':(['ALA','CYS'],[[101,1,2],[102,2,2]]),'L':(['ASP','GLU'],[[99,1,2],[98,2,2]])}
        def write(path,rigid=False,displace=False):
            s=Structure.Structure('m');m=Model.Model(0);s.add(m);serial=0
            rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
            for name,(letters,coords) in chains.items():
                c=Chain.Chain(name);m.add(c)
                for i,(resname,coord) in enumerate(zip(letters,coords),1):
                    pos=np.asarray(coord,dtype=float)
                    if displace and name in ('A','B'):pos+=np.array([100,0,0])
                    if rigid:pos=pos@rotation+np.array([100,20,5])
                    residue=Residue.Residue((' ',i,' '),resname,'');c.add(residue);serial+=1
                    residue.add(Atom.Atom('CA',pos,20,1,' ',' CA ',serial,element='C'))
            writer=MMCIFIO();writer.set_structure(s);writer.save(str(path))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ref=root/'ref.cif';pred=root/'pred.cif';write(ref);write(pred,rigid=True)
            result=audit(ref,pred)
            self.assertLess(result['antigen_aligned_fv_ca_rmsd'],1e-5)
            self.assertEqual(result['fraction_native_contacts'],1)
            write(pred,displace=True);result=audit(ref,pred)
            self.assertGreater(result['antigen_aligned_fv_ca_rmsd'],99)
            self.assertEqual(result['fraction_native_contacts'],0)
            with self.assertRaisesRegex(ValueError,'biological copy'):audit(ref,pred,reference_chains=('H','L','Y'))


if __name__=='__main__':unittest.main()
