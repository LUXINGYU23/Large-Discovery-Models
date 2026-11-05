import tempfile
import unittest
from pathlib import Path
from ldm_tts.engine.run_store import BudgetLedger,BudgetExceededError
from tasks.copper_inversion.v2.budget import ScientificSlots,fully_accounted


class ScientificBudget(unittest.TestCase):
    def test_full_cap_requires_outcome_for_every_source(self):
        from types import SimpleNamespace
        ledger=BudgetLedger(limits={"algorithm_feedback_slots":3})
        slots=ScientificSlots(ledger.consume_many)
        slots.reserve("rejected",interface_rejected=True)
        slots.reserve("done")
        observations=[SimpleNamespace(canonical_key="done")]
        self.assertFalse(fully_accounted(ledger,observations))
        slots.reserve("inflight")
        self.assertFalse(fully_accounted(ledger,observations))
        observations.append(SimpleNamespace(canonical_key="inflight"))
        self.assertTrue(fully_accounted(ledger,observations))
        observations.append(SimpleNamespace(canonical_key="unpaid"))
        self.assertFalse(fully_accounted(ledger,observations))

    def test_engine_counts_scientific_rejections_in_remaining_budget(self):
        from types import SimpleNamespace
        from ldm_tts.engine import LDMEngineState,LDMEngineConfig
        from ldm_tts.engine.evaluation import remaining_fresh_attempts,evaluation_budget_stop_reason
        ledger=BudgetLedger(limits={"algorithm_feedback_slots":3})
        runtime=SimpleNamespace(budget=ledger);state=LDMEngineState()
        config=LDMEngineConfig(iterations=20,reservoir_size=32,max_evaluation_attempts=321)
        ScientificSlots(ledger.consume_many).reserve("rejected",interface_rejected=True)
        self.assertEqual(remaining_fresh_attempts(runtime,state,config),2)
        for key in ['a','b']:ScientificSlots(ledger.consume_many).reserve(key,interface_rejected=True)
        self.assertEqual(remaining_fresh_attempts(runtime,state,config),0)
        self.assertEqual(evaluation_budget_stop_reason(runtime,state,config),'external_evaluation_budget')

    def test_rejected_and_evaluated_share_cap_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(limits={"algorithm_feedback_slots":2},path=Path(tmp)/"budget.json")
            slots=ScientificSlots(ledger.consume_many)
            slots.reserve("invalid",interface_rejected=True)
            slots.reserve("invalid",interface_rejected=True)
            slots.reserve("valid")
            slots.reserve("valid")
            self.assertEqual(ledger.counters["algorithm_feedback_slots"],2)
            self.assertEqual(ledger.counters["interface_rejected_sources"],1)
            with self.assertRaises(BudgetExceededError):slots.reserve("new")
            restored=BudgetLedger.load(ledger.path)
            ScientificSlots(restored.consume_many).reserve("valid")
            self.assertEqual(restored.counters["algorithm_feedback_slots"],2)


if __name__=="__main__":unittest.main()
