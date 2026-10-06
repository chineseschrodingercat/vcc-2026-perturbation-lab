import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('restore_assets',Path(__file__).resolve().parents[1]/'scripts/restore_model_assets.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class RestoreAssetsTests(unittest.TestCase):
    def archive(self,folder,path='models/example.pt',payload=b'synthetic-model-test',expected=None):
        sha=hashlib.sha256(payload if expected is None else expected).hexdigest()
        blob='blobs/'+sha+'.pt';p=folder/'model.zip'
        with zipfile.ZipFile(p,'w') as z:
            z.writestr(blob,payload)
            z.writestr('RESTORE_MANIFEST.json',json.dumps({'schema_version':1,'family':'test','assets':[{'path':path,'sha256':sha,'bytes':len(payload),'role':'model-asset','blob':blob}]}))
        return p

    def test_restore_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d);p=self.archive(f);w=f/'workspace'
            self.assertEqual(module.restore(p,w)['created'],1)
            self.assertEqual(module.restore(p,w)['already_identical'],1)

    def test_different_destination_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d);p=self.archive(f);w=f/'workspace';dest=w/'models/example.pt';dest.parent.mkdir(parents=True);dest.write_bytes(b'keep')
            with self.assertRaises(ValueError):module.restore(p,w)
            self.assertEqual(dest.read_bytes(),b'keep')

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d);p=self.archive(f,path='../outside.pt')
            with self.assertRaises(ValueError):module.restore(p,f/'workspace')

    def test_corrupt_blob_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d);p=self.archive(f,expected=b'expected-but-corrupt')
            with self.assertRaises(ValueError):module.restore(p,f/'workspace')
            self.assertFalse((f/'workspace').exists())

if __name__=='__main__':unittest.main()
