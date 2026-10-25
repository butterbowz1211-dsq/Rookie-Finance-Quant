"""Protect replication counts against accidental duplicate Monte Carlo records."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'research'/'ML金融学习计划'/'A_蒙特卡洛'
sys.path.insert(0,str(SOURCE))
import table_a3
import table_a4

class SimulationRecordTests(unittest.TestCase):
    def test_a3_resume_checks_signature_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.jsonl'
            signature={'pc':50}
            record={'seed':0,'method':'Lasso','run_signature':signature,'sel':[1]}
            path.write_text(json.dumps(record)+'\n',encoding='utf-8')
            self.assertEqual(table_a3.read_completed(path,signature),{0:{'Lasso'}})
            with self.assertRaises(ValueError): table_a3.read_completed(path,{'pc':100})
            path.write_text((json.dumps(record)+'\n')*2,encoding='utf-8')
            with self.assertRaises(ValueError): table_a3.read_completed(path,signature)

    def test_existing_a3_and_a4_records_require_explicit_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for module,prefix in [(table_a3,'a3'),(table_a4,'a4')]:
                (out/f'raw_{prefix}_example.jsonl').write_text('{}\n',encoding='utf-8')
                with patch.object(module,'OUT',out), patch.object(sys,'argv',['runner','--tag','example']):
                    with self.assertRaises(FileExistsError): module.main()

if __name__=='__main__': unittest.main()
