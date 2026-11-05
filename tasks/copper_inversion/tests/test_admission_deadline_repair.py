"""Concurrency and real POSIX signal regression; execute only on the remote host."""
import ast
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.v2.interface_checks import InterfaceChecks

class RepairTests(unittest.TestCase):
    def test_all_six_fixtures_overlap_and_inputs_are_prepared_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=InterfaceChecks(tmp,Mock());p=seal(Path(tmp),b'def solve(problem,budget,seed): return None\n')
            barrier=threading.Barrier(6)
            calls=[]
            def work(payload,source,problem):barrier.wait(timeout=5);calls.append(problem['case_id'])
            c._check_fixture=work
            self.assertTrue(c(p));self.assertEqual(len(calls),6)
            files=list((Path(tmp)/'interface-inputs').glob('*/*'));before={str(f):f.stat().st_mtime_ns for f in files}
            self.assertTrue(c(p));self.assertEqual(before,{str(f):f.stat().st_mtime_ns for f in files})
    def test_parallel_failures_report_fixture_order_and_charge_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            slots=Mock();c=InterfaceChecks(tmp,Mock(),slots);p=seal(Path(tmp),b'def solve(problem,budget,seed): return None\n')
            def work(payload,source,problem):
                if problem['case_id'].endswith('0'):time.sleep(.03);raise ValueError('first')
                raise ValueError('later')
            c._check_fixture=work
            with self.assertRaisesRegex(ValueError,'first'):c(p)
            slots.reserve.assert_called_once_with(p['sha256'],interface_rejected=True)
    def deadline_runner(self):
        # Extract the exact production exception/cleanup structure; replace only
        # the numerical search with a short real one-shot alarm workload.
        path=Path(__file__).parents[1]/'experiments/v2/stochastic_solver.py'
        tree=ast.parse(path.read_text());solve=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='solve')
        guarded=next(n for n in solve.body if isinstance(n,ast.Try))
        guarded.body[0].body=ast.parse("signal.setitimer(signal.ITIMER_REAL,.00001)\nfor i in range(80): pass").body
        definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('SearchDeadline','stop_search')]
        func=ast.parse("def run(previous_handler): pass").body[0];func.body=[guarded]
        module=ast.fix_missing_locations(ast.Module(body=definitions+[func],type_ignores=[]))
        ns={'signal':signal};exec(compile(module,str(path),'exec'),ns);return ns
    def test_pending_alarm_at_cleanup_entry_is_consumed(self):
        n=self.deadline_runner();real=signal.signal;calls=[]
        def interrupted(sig,handler):
            calls.append(handler)
            if len(calls)==1:raise n['SearchDeadline']()
            return real(sig,handler)
        old=signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM,n['stop_search'])
        with patch.object(signal,'signal',side_effect=interrupted), patch.object(signal,'setitimer'):
            n['run'](old)
        self.assertEqual(signal.getsignal(signal.SIGALRM),old)
    def test_real_one_shot_alarm_cleanup_stress(self):
        n=self.deadline_runner();old=signal.getsignal(signal.SIGALRM)
        for _ in range(10000):
            signal.signal(signal.SIGALRM,n['stop_search']);n['run'](old)
            self.assertEqual(signal.getsignal(signal.SIGALRM),old)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)

if __name__=='__main__':unittest.main()
