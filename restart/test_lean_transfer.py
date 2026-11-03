"""Small semantic checks; these do not establish biological performance."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

import lean_transfer as lt


class LeanTransferTests(unittest.TestCase):
    def test_raw_construct_means_use_weights_and_exclude_noncore_ntcs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=ad.AnnData(np.asarray([[.5,2],[2.5,4],[1,1],[900,900]],np.float32),
                obs=pd.DataFrame({"num_cells_filtered":[1,3,8,9],"core_control":[False,False,True,False]},
                    index=["0_T_P1_ENSG0001","1_T_ENST00000421176.3,ENST00000602553.1_ENSG0001","9_non-targeting_non-targeting_non-targeting","10_non-targeting_non-targeting_non-targeting"]),
                var=pd.DataFrame({"gene_name":["G1","G2"]},index=["ENSG1","ENSG2"]))
            a.write_h5ad(root/"bulk.h5ad")
            args=lt.parser().parse_args(["ingest","--input",str(root/"bulk.h5ad"),"--output",str(root/"bulk.npz"),
                "--source-id","k1","--background","K562","--kind","raw-bulk","--target-col","construct-index",
                "--n-col","num_cells_filtered","--control-col","core_control"])
            args.control_label=["non-targeting"]
            with contextlib.redirect_stdout(io.StringIO()):
                lt.ingest(args)
            source=lt.load_source(root/"bulk.npz")
            np.testing.assert_allclose(source.target_mean,[[2,3.5]])
            np.testing.assert_allclose(source.control_mean,[1,1])
            self.assertEqual(source.targets.tolist(),["T"])
            self.assertEqual(source.n_cells.tolist(),[4])
            self.assertEqual(source.control_n,8)
            self.assertEqual(json.loads((root/"bulk.json").read_text())["excluded_noncore_control_rows"],1)

    def test_sparse_cells_are_aggregated_and_fractional_cells_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=ad.AnnData(sparse.csr_matrix([[1,2],[3,0],[4,6]],dtype=np.float32),
                obs=pd.DataFrame({"gene":["NTC","T","T"]},index=["a","b","c"]),
                var=pd.DataFrame(index=["G1","G2"]))
            a.write_h5ad(root/"cells.h5ad")
            args=lt.parser().parse_args(["ingest","--input",str(root/"cells.h5ad"),"--output",str(root/"cells.npz"),
                "--source-id","s","--background","B","--kind","singlecell","--chunk-rows","2"])
            args.control_label=["NTC"]
            with contextlib.redirect_stdout(io.StringIO()):
                lt.ingest(args)
            source=lt.load_source(root/"cells.npz")
            np.testing.assert_allclose(source.target_mean,[[3.5,3]])
            self.assertEqual(source.n_cells.tolist(),[2])
            with self.assertRaises(ValueError):
                lt.check_raw(np.array([[.1]]),True)

    @staticmethod
    def source(name,background,means,control=(10.,20.,30.)):
        return lt.Source(name,background,np.array(["G1","G2","G3"]),np.array(["T1","T2"]),
                          np.asarray(means,dtype=float),np.asarray(control),np.array([100.,100.]),500,np.ones(3,bool))

    def test_whole_background_exclusion_and_one_vote(self):
        k1=self.source("k1","K562",[[20,10,30],[10,25,25]])
        k2=self.source("k2","K562",[[100,10,30],[10,100,25]])
        k2.n_cells=np.array([30.,30.])
        r=self.source("r","RPE1",[[14,15,31],[11,22,27]])
        heldout=[s for s in [k1,k2,r] if s.background=="K562"]
        train=[s for s in [k1,k2,r] if s.background!="K562"]
        genes=lt.common_genes([k1,k2,r],minimum=3)
        query,_=lt.representative_control(heldout,genes)
        before,audit=lt.predict_delta(train,genes,["T1","T2"],query)
        k1.target_mean[:]=99999
        k2.target_mean[:]=99999
        after,_=lt.predict_delta(train,genes,["T1","T2"],query)
        np.testing.assert_array_equal(before,after)
        self.assertEqual([r["background"] for r in audit["backgrounds"]],["RPE1"])
        k1.target_mean=np.array([[20,10,30],[10,25,25]],float)
        single,_=lt.predict_delta([k1,r],genes,["T1","T2"],query)
        duplicate,audit=lt.predict_delta([k1,k2,r],genes,["T1","T2"],query)
        np.testing.assert_allclose(single,duplicate)
        self.assertEqual(len(audit["backgrounds"]),2)

    def test_zero_projection_and_missing_measurement(self):
        a=self.source("a","A",[[20,10,30],[10,25,25]])
        b=self.source("b","B",[[20,10,30],[10,25,25]])
        b.measured[2]=False;b.__post_init__()
        self.assertEqual(lt.common_genes([a,b],minimum=2).tolist(),["G1","G2"])
        delta,means=lt.project_delta(np.zeros((2,3)),np.array([10.,20.,30.]))
        np.testing.assert_array_equal(delta,np.zeros((2,3)))
        np.testing.assert_array_equal(means,np.array([[10,20,30],[10,20,30]]))

    def test_stream_writer_labels_integer_counts_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"tiny.h5ad"
            contexts=["D","E","F"];targets=["T1","T2"];genes=["G1","G2","G3"]
            lt.init_output(path,contexts,targets,genes,cells_per_target=4)
            nnz=offset=0
            with h5py.File(path,"r+") as f:
                for i in range(6):
                    a=lt.sample_counts(np.array([3.,7.,0.]),np.array([3.,7.,0.]),np.array([6.,20.,0.]),
                                       np.random.default_rng(i),n=4)
                    b=lt.sample_counts(np.array([3.,7.,0.]),np.array([3.,7.,0.]),np.array([6.,20.,0.]),
                                       np.random.default_rng(i),n=4)
                    np.testing.assert_array_equal(a.toarray(),b.toarray())
                    nnz=lt.append_block(f["X"],a,offset,nnz);offset+=4
            a=ad.read_h5ad(path)
            self.assertEqual(a.shape,(24,3))
            self.assertEqual(a.obs.context.astype(str).unique().tolist(),contexts)
            self.assertTrue((a.obs.groupby(["context","target_gene"],observed=True).size()==4).all())
            self.assertTrue(np.issubdtype(a.X.dtype,np.integer))
            self.assertTrue((a.X.data>0).all())

    def test_parallel_counts_match_serial_and_prefetch_is_bounded(self):
        targets=[f"T{i}" for i in range(31)]
        mean=np.array([8.,12.,4.,0.,5.])
        variance=np.array([24.,36.,8.,0.,5.])
        positions=np.array([0,2])
        desired=np.array([[9.+i/10,3.+i/20] for i in range(len(targets))])
        common=dict(seed=20261103,context_index=2,on_target_fraction=.2,n=8)
        gene_positions={"T0":0,"T1":1}
        for generator in ("nb","poisson"):
            expected=list(lt.sample_target_blocks(targets,mean,variance,positions,desired,gene_positions,
                          generator=generator,workers=1,**common))
            offered=0
            def lazy_targets():
                nonlocal offered
                for target in targets:
                    offered+=1
                    yield target
            with contextlib.closing(lt.sample_target_blocks(lazy_targets(),mean,variance,positions,desired,
                                    gene_positions,generator=generator,workers=6,**common)) as blocks:
                for delivered,(ti,target,block) in enumerate(blocks):
                    # No eager submission of all 31 jobs; the current yielded
                    # block is included in this 12-job prefetch window.
                    self.assertLessEqual(offered-delivered,12)
                    self.assertEqual((ti,target),expected[delivered][:2])
                    serial=expected[delivered][2]
                    np.testing.assert_array_equal(block.data,serial.data)
                    np.testing.assert_array_equal(block.indices,serial.indices)
                    np.testing.assert_array_equal(block.indptr,serial.indptr)
            self.assertEqual(offered,len(targets))

    def test_csv_axis_handles_header_and_no_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"genes.csv"
            path.write_text("gene_name\nA\nB\n")
            self.assertEqual(lt.read_axis(path,{"gene_name"}).tolist(),["A","B"])
            path.write_text("A\nB\n")
            self.assertEqual(lt.read_axis(path,{"gene_name"}).tolist(),["A","B"])

    def test_source_locks_reject_same_id_changed_background_or_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"source.npz"
            def write_source(background="K562", value=10.):
                np.savez_compressed(path,source_id=np.asarray("same_id"),background=np.asarray(background),
                    gene_names=np.asarray(["G1","G2"]),target_names=np.asarray(["T1"]),
                    target_mean_counts=np.asarray([[value,20.]]),control_mean_counts=np.asarray([10.,20.]),
                    target_n_cells=np.asarray([100.]),control_n_cells=np.asarray(500.),measured_mask=np.ones(2,bool))
            write_source()
            sources=lt.load_sources([path])
            model={"source_locks":lt.source_locks(sources)}
            lt.validate_source_locks(sources,model)
            with self.assertRaisesRegex(ValueError,"Duplicate source_id"):
                lt.load_sources([path,path])
            with self.assertRaisesRegex(ValueError,"no source content locks"):
                lt.validate_source_locks(sources,{})
            write_source(background="RPE1")
            with self.assertRaisesRegex(ValueError,"Source lock mismatch"):
                lt.validate_source_locks(lt.load_sources([path]),model)
            write_source(background="K562",value=999.)
            with self.assertRaisesRegex(ValueError,"Source lock mismatch"):
                lt.validate_source_locks(lt.load_sources([path]),model)

    def test_count_bounds_before_integer_narrowing(self):
        with self.assertRaises(ValueError):
            lt.sample_counts(np.zeros(2),np.ones(2),np.ones(2),np.random.default_rng(1))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"bounds.h5ad"
            lt.init_output(path,["D"],["T"],["G1","G2"],cells_per_target=2)
            with h5py.File(path,"r+") as f:
                for invalid in (np.array([[1.9,0],[0,2]]),np.array([[0,0],[0,2]]),
                                np.array([[600_000,600_000],[0,2]]),np.array([[2**32+1,0],[0,2]],dtype=np.int64)):
                    with self.assertRaises(ValueError):
                        lt.append_block(f["X"],sparse.csr_matrix(invalid),0,0)
                self.assertEqual(len(f["X/data"]),0)
                end=lt.append_block(f["X"],sparse.csr_matrix([[1_000_000,0],[0,1]],dtype=np.int64),0,0)
                self.assertEqual(end,2)
                np.testing.assert_array_equal(f["X/data"][:],[1_000_000,1])

    def test_missing_context_column_requires_manifest_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"context_D.h5ad"
            a=ad.AnnData(sparse.csr_matrix([[1,2],[3,4]]),
                obs=pd.DataFrame({"target_gene":["non-targeting","non-targeting"]},index=["a","b"]),
                var=pd.DataFrame(index=["G1","G2"]))
            a.write_h5ad(path)
            with self.assertRaisesRegex(ValueError,"manifest-to-file binding"):
                lt.control_statistics(path,"D",["G1","G2"])
            audit={}
            mean,var,n,columns=lt.control_statistics(path,"D",["G1","G2"],manifest_binding={"D":path},label_audit=audit)
            np.testing.assert_array_equal(mean,[2,3])
            self.assertIn("obs.context absent",audit["context_label_source"])
            a.obs["context"]="E";a.write_h5ad(path)
            with self.assertRaisesRegex(ValueError,"do not retain manifest context"):
                lt.control_statistics(path,"D",["G1","G2"],manifest_binding={"D":path})


if __name__=="__main__":
    unittest.main()
