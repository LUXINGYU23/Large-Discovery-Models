"""Shared discovery allowance for rejected sources and completed suite attempts."""


def fully_accounted(ledger, observations):
    """A full source cap is terminal only when every paid source has an outcome."""
    prefix = "copper:v2:scientific-source:"
    records = {key[len(prefix):]: usage for key, usage in
               ledger.metadata.get("cumulative_usage", {}).items() if key.startswith(prefix)}
    cap = ledger.limits.get("algorithm_feedback_slots")
    if cap is None or len(records) != cap or ledger.counters.get("algorithm_feedback_slots") != cap:
        return False
    if any(usage.get("algorithm_feedback_slots") != 1 for usage in records.values()):
        raise ValueError("non-unit scientific source charge")
    completed = {o.canonical_key for o in observations}
    rejected = {key for key, usage in records.items() if usage.get("interface_rejected_sources") == 1}
    return completed <= records.keys() and set(records) == completed | rejected

class ScientificSlots:
    def __init__(self, account):
        self.account = account

    def reserve(self, source_sha256, *, interface_rejected=False):
        amounts = {"algorithm_feedback_slots": 1}
        if interface_rejected:
            amounts["interface_rejected_sources"] = 1
        return self.account(amounts, usage_key="copper:v2:scientific-source:" + source_sha256)
