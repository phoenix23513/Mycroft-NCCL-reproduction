"""Real source preparation and simulated rank coordination; no GPU or CUDA claim."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"cluster/crater/scripts"))
import package_m3
import prepare_m3
import run_m3


class SourceDeliveryTests(unittest.TestCase):
    def test_real_two_layer_patch_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix="m3-source-") as temp:
            root=Path(temp);archive=root/"upload.tar.gz";package_m3.package(archive)
            with tarfile.open(archive) as tar:
                for member in tar:
                    self.assertTrue(member.isfile())
                    path=root/member.name;path.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as source:path.write_bytes(source.read())
            bundle=root/"m3-experiment";dest=root/"source"
            proof=prepare_m3.prepare(bundle,dest)
            self.assertEqual(len(proof["inputs"]),9);self.assertEqual(len(proof["patched_files"]),12)
            text=(dest/"src/transport/net.cc").read_text()
            self.assertIn('if (mycroftM3AllowSend(args, sub)) {',text)
            self.assertLess(text.index('mycroftM2Ready(args, sub)'),text.index('mycroftM3AllowSend(args, sub)'))
            self.assertIn('mycroftM3AllowSend',text)
            self.assertIn('// Check whether the network has completed some send operations.',text)
            for source,target in prepare_m3.COPIES.items():
                self.assertEqual((bundle/source).read_bytes(),(dest/target).read_bytes())
            self.assertEqual(subprocess.check_output(['git','-C',str(ROOT/'third_party/nccl'),'status','--porcelain']),b'')
            with self.assertRaises(FileExistsError):prepare_m3.prepare(bundle,dest)
            input_path=bundle/(prepare_m3.M3+'include/send_delay.h')
            input_path.write_bytes(input_path.read_bytes()+b'// tamper\n')
            with self.assertRaisesRegex(ValueError,'differs'):prepare_m3.prepare(bundle,root/'tampered')
            self.assertFalse((root/'tampered').exists())


class SuiteCoordinationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='m3-suite-control-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.directory=self.root/'suite'
        self.baseline=self.root/'baseline';(self.baseline/'lib').mkdir(parents=True)
        (self.baseline/'lib/libnccl.so.2.21.5').write_bytes(b'fake CPU fixture, not NCCL')
        self.digest=hashlib.sha256(b'fake CPU fixture, not NCCL').hexdigest()
        self.target={'rank':0,'operation':6,'channel':0,'delay_ns':100000000}
        self.observed=[];self.fail_case=None

    def execute_fixture(self,rank,directory,product,config,timeout,*,capture_m2=False,delay_target=None):
        folder=directory/f'rank{rank}';folder.mkdir()
        self.observed.append((directory.name,rank,capture_m2,delay_target))
        run_m3.write_json(folder/'ready.json',{'run_id':config['run_id']})
        peer=run_m3.wait_json(directory/f'rank{1-rank}/ready.json',timeout)
        self.assertEqual(peer['run_id'],config['run_id'])
        code=17 if directory.name==self.fail_case and rank==1 else 0
        run_m3.write_json(folder/'status.json',{'run_id':config['run_id'],'exit_code':code})
        return code,False

    def verify_fixture(self,arguments,log,timeout,environment=None):
        log.write_text('SIMULATED_VERIFIER_FOR_COORDINATION_TEST_ONLY\n');return 0

    def run_pair(self):
        with patch.object(run_m3,'check_bundle',return_value={}),patch.object(run_m3,'product_proof',return_value=({},'f'*64)),\
             patch.object(run_m3,'execute',side_effect=self.execute_fixture),patch.object(run_m3,'run',side_effect=self.verify_fixture),\
             redirect_stdout(io.StringIO()):
            with ThreadPoolExecutor(max_workers=2) as pool:
                # Worker-first rendezvous is supported by the actual shared-directory code.
                worker=pool.submit(run_m3.run_suite,1,self.directory,self.baseline,self.root/'product',self.digest,self.target,2)
                master=pool.submit(run_m3.run_suite,0,self.directory,self.baseline,self.root/'product',self.digest,self.target,2)
                return master.result(timeout=10),worker.result(timeout=10)

    def test_three_cases_one_bundle_correct_switches_and_unique_ids(self):
        self.assertEqual(self.run_pair(),(0,0))
        ids=[]
        for name in run_m3.CASES:
            config=json.loads((self.directory/name/'run-config.json').read_text());ids.append(config['run_id'])
            expected=None if name=='baseline' else dict(self.target,delay_ns=0 if name=='capture' else self.target['delay_ns'])
            for rank in (0,1):self.assertIn((name,rank,name!='baseline',expected),self.observed)
        self.assertEqual(len(set(ids)),3)
        archive=self.directory.with_name('suite.tar.gz')
        with tarfile.open(archive) as tar:
            self.assertIn('m3-results/delay/rank1/status.json',tar.getnames())
            self.assertNotIn('m3-results/baseline/rank0/capture/injection.json',tar.getnames())
        before=archive.read_bytes()
        with patch.object(run_m3,'check_bundle',return_value={}),patch.object(run_m3,'product_proof',return_value=({},'f'*64)):
            with self.assertRaisesRegex(ValueError,'archive exists'):
                run_m3.run_suite(0,self.directory,self.baseline,self.root/'product',self.digest,self.target,2)
            with self.assertRaises(FileExistsError):
                run_m3.run_suite(1,self.directory,self.baseline,self.root/'product',self.digest,self.target,2)
        self.assertEqual(archive.read_bytes(),before)

    def test_failed_case_stops_and_packages_existing_evidence(self):
        self.fail_case='baseline'
        self.assertEqual(self.run_pair(),(1,1))
        self.assertFalse((self.directory/'capture').exists());self.assertFalse((self.directory/'delay').exists())
        with tarfile.open(self.directory.with_name('suite.tar.gz')) as tar:
            status=tar.extractfile('m3-results/baseline/run-status.txt').read().decode()
            self.assertIn('rank1_exit_code=17',status)
            manifest=json.loads(tar.extractfile('m3-results/bundle-manifest.json').read())
            self.assertIn('delay/rank0/status.json',manifest['missing_files'])


if __name__=='__main__':unittest.main()
