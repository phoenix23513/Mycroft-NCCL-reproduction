"""Handwritten injection windows test rejection rules, not actual M3 execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

M3=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("m3_evidence",M3/"verify_experiment.py")
VERIFY=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(VERIFY)


class InjectionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="m3-injection-fixture-");self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.capture=self.folder/"capture"
        (self.capture/"trace").mkdir(parents=True)
        self.target={"rank":0,"operation":6,"channel":0,"delay_ns":100}
        self.item={"injection_version":1,"clock":"CLOCK_MONOTONIC","kind":"software_isend_delay",
                   "local_rank":0,"target_rank":0,"target_operation":6,"target_channel":0,"delay_ns":100,
                   "comm_hash":42,"connection_id":1,"held_step":0,"begin_ns":100,"deadline_ns":200,
                   "release_ns":210,"blocked_attempts":3,"started":True,"released":True,"error":False}
        common={"op_seq_candidate":6,"channel":0,"comm_hash":42,"connection_id":1,
                "gpu_ready_steps":2,"nsteps":8,"transmitted_steps":0,"done_steps":0}
        self.rows=[dict(common,observed_at_ns=t) for t in (101,121,191)]
        self.rows.append(dict(common,observed_at_ns=250,transmitted_steps=8,done_steps=8))
        self.completion={"record_kind":"operation_completion","op_seq_candidate":6,"observed_at_ns":260}
        self.save()

    def save(self):
        (self.capture/"injection.json").write_text(json.dumps(self.item))
        (self.capture/"capture-manifest.json").write_text(json.dumps({"sample_period_ns":10}))
        (self.capture/"trace/send-rank0.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        (self.capture/"trace/plan-map.jsonl").write_text(json.dumps(self.completion)+'\n')

    def verify(self):return VERIFY.verify_injection(self.folder,0,self.target)

    def test_sampled_hold_and_recovery(self):
        result=self.verify();self.assertEqual(result["hold_samples"],3)
        self.assertEqual(result["observed_hold_ns"],110)
        self.assertEqual(result["max_sample_gap_ns"],70)

    def test_configured_but_unhit_injection_rejected(self):
        self.item["started"]=False;self.save()
        with self.assertRaisesRegex(ValueError,"not hit"):self.verify()

    def test_early_release_rejected(self):
        self.item["release_ns"]=199;self.save()
        with self.assertRaisesRegex(ValueError,"before deadline"):self.verify()

    def test_submission_during_hold_rejected(self):
        self.rows[1]["transmitted_steps"]=2;self.save()
        with self.assertRaisesRegex(ValueError,"not-submitted"):self.verify()

    def test_no_periodic_window_rejected(self):
        self.rows=self.rows[:1]+self.rows[-1:];self.save()
        with self.assertRaisesRegex(ValueError,"not-submitted"):self.verify()

    def test_off_case_does_not_accept_active_injection(self):
        self.target["delay_ns"]=0;self.item["delay_ns"]=0;self.save()
        with self.assertRaisesRegex(ValueError,"non-target"):self.verify()

    def test_completion_before_release_rejected(self):
        self.completion["observed_at_ns"]=209;self.save()
        with self.assertRaisesRegex(ValueError,"completion precedes"):self.verify()


if __name__=="__main__":unittest.main()
