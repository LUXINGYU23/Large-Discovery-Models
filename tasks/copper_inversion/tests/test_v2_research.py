"""Public research export and fixed configuration checks; remote only."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from tasks.copper_inversion.v2.research import PublicTrainingExport,prepare_resources
from tasks.copper_inversion.v2.workflow import parse_args


class ResearchV2(unittest.TestCase):
    def test_registered_entry_is_v2_and_rejects_legacy_switches(self):
        from tasks.copper_inversion.ldm_task import procedure
        from tasks.copper_inversion.v2 import workflow
        self.assertIs(procedure.main,workflow.main)
        required=["--public-catalog","/unused","--evaluation-socket","/unused","--evaluation-contract","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","--evaluation-service-uid","991","--out-dir","/unused","--embedding-model-path","/unused","--harness-cache-dir","/unused"]
        for obsolete in (["--encoding","ast"],["--encoding","qwen"],["--case-id","drillholes_3d"],["--mock"]):
            with self.assertRaises(SystemExit):procedure.parse_args(required+obsolete)

    def test_fixed_fusion_provider(self):
        a=parse_args(["--public-catalog","/unused","--evaluation-socket","/unused","--evaluation-contract","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","--evaluation-service-uid","991","--out-dir","/unused","--embedding-model-path","/unused","--harness-cache-dir","/unused"])
        self.assertEqual((a.encoding,a.llm_url,a.llm_model,a.thinking),("fusion","https://api.deepseek.com","deepseek-flash","low"))
        self.assertEqual(1+a.iterations*a.evaluations_per_round,321)

    def test_development_export_allowlist(self):
        manifest=Path("public-training-integration-r1/catalog.json")
        if not manifest.exists():self.skipTest("requires remote integration suite")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);export=PublicTrainingExport(manifest)
            export(root,{})
            public=root/"public"
            self.assertEqual(len(json.loads((public/"cases.json").read_text())),2)
            self.assertFalse(list(public.rglob("truth.npy")))
            self.assertFalse(list(public.rglob("*manifest*")))
            self.assertFalse((public/"tasks/copper_inversion/v2/metrics.py").exists())
            self.assertFalse((public/"tasks/copper_inversion/v2/suite.py").exists())
            resources=prepare_resources(root)
            self.assertIn("cases.json",(resources/"tools/copper.mjs").read_text())
            self.assertNotIn("objective_v1",(public/"TASK.md").read_text())
            private="parent_id"
            for path in public.rglob("*.json"):
                self.assertNotIn(private,path.read_text())
            from tasks.copper_inversion.core.harness_runtime import create_client
            a=parse_args(["--public-catalog",str(manifest),"--evaluation-socket","/unused",
                "--evaluation-contract","a"*64,"--evaluation-service-uid","991",
                "--out-dir",str(root),"--embedding-model-path","/unused","--harness-cache-dir",str(root/'cache')])
            a.case_id='development';a._prepare_public=export;a._resources=resources
            with patch('tasks.copper_inversion.core.harness_runtime.api_key',return_value='test-canary'):
                client=create_client(a,root,{})
            self.assertEqual(client.config.network_policy.denied_hosts,('*',))
            self.assertEqual(client.config.base_url,'https://api.deepseek.com')


if __name__=="__main__":unittest.main()
