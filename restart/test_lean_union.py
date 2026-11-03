import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import lean_transfer as lt
import lean_union as lu


class UnionTests(unittest.TestCase):
    @staticmethod
    def source(name,background,genes,control,means):
        return lt.Source(name,background,np.asarray(genes),np.asarray(["T1","T2"]),np.asarray(means,float),
                         np.asarray(control,float),np.array([100.,100.]),500,np.ones(len(genes),bool))

    def test_missing_measurement_does_not_dilute_other_background(self):
        k=self.source("k","K562",["G1","G2","G3"],[10,20,5],[[20,10,20],[15,15,7]])
        r=self.source("r","RPE1",["G1","G2","G4"],[10,20,8],[[10,20,8],[15,15,8]])
        core=np.asarray(["G1","G2"]);genes=np.asarray(["G1","G2","G3","G4"])
        query=np.array([10.,20.,5.,8.])
        pred,mask,audit=lu.predict_delta([k,r],core,genes,["T1"],query,alpha=1,temperature=0,prior_cells=0)
        expected=np.log1p(lt.SCALE*20/30)-np.log1p(lt.SCALE*5/30)
        self.assertAlmostEqual(pred[0,2],expected,places=6)
        self.assertTrue(mask[0,2]);self.assertTrue(mask[0,3])
        self.assertEqual(len(audit["backgrounds"]),2)

    def test_unobserved_union_gene_preserves_control(self):
        k=self.source("k","K562",["G1","G2","G3"],[10,20,5],[[20,10,20],[15,15,7]])
        core=np.asarray(["G1","G2"]);genes=np.asarray(["G1","G2","G3","G4"])
        query=np.array([10.,20.,5.,9.])
        pred,mask,_=lu.predict_delta([k],core,genes,["T1"],query,alpha=.5)
        actual,wanted=lu.project(pred,query,[0,1],mask)
        self.assertFalse(mask[0,3]);self.assertEqual(wanted[0,3],9)
        self.assertAlmostEqual(wanted[0,:2].sum(),30,places=8)

    def test_loco_cv_artifacts_and_background_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            sources=[self.source("k1","K562",["G1","G2","G3"],[10,20,5],[[20,10,20],[15,15,7]]),
                     self.source("k2","K562",["G1","G2","G3"],[10,20,5],[[20,10,20],[15,15,7]]),
                     self.source("r","RPE1",["G1","G2","G4"],[10,20,8],[[10,20,8],[15,15,8]])]
            paths=[]
            for s in sources:
                p=root/f"{s.source_id}.npz";paths.append(str(p))
                np.savez_compressed(p,source_id=np.asarray(s.source_id),background=np.asarray(s.background),
                    gene_names=s.genes,target_names=s.targets,target_mean_counts=s.target_mean,
                    control_mean_counts=s.control_mean,target_n_cells=s.n_cells,
                    control_n_cells=np.asarray(s.control_n),measured_mask=s.measured)
            args=SimpleNamespace(sources=paths,genes=None,min_genes=2,min_cells=30,max_targets=2,prior_cells=50,
                                 alphas="0.5",temperatures="0",seed=1,output_dir=str(root/"run"))
            with contextlib.redirect_stdout(io.StringIO()):
                lu.cv(args)
            report=json.loads((root/"run/cv_report.json").read_text())
            self.assertEqual(report["common_normalization_genes"],2)
            self.assertEqual(report["union_output_genes"],4)
            fold=next(f for f in report["folds"] if f["heldout_background"]=="K562")
            self.assertEqual(set(fold["excluded_sources"]),{"k1","k2"})
            self.assertEqual(fold["training_sources"],["r"])
            self.assertTrue((root/"run/selected_model.json").exists())


if __name__=="__main__":
    unittest.main()
