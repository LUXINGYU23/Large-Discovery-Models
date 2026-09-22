"""The pinned native generation loop with metered model/check callbacks."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
import typing
import uuid

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.data.ir import make_complete_design_ir
from ldm_tts.engine.run_store import BudgetExceededError, atomic_json_write
from .generator import Generator
from .grammar import REGISTRY
from .native_runtime import NativeStop
from .protocol import digest
from .receipts import Receipts


SOURCES = {
    "agent/generator_qlib_search.py": "45c7e2aabc7511c9a91b37bedda3eead048f9e6aeec7a61f96fffe5c585b2f9b",
    "agent/prompts_qlib_instruction.py": "35ae92777a219b324c286fe5dd6b117cae6fa749f2b8d0b14942000012f7b6f6",
}


def generator_source(source):
    names = {"get_system_searcher_prompt", "_normalize_llm_output", "_unique_name", "call_qlib_search"}
    nodes = [node for node in ast.parse(source).body if
             isinstance(node, ast.FunctionDef) and node.name in names or
             isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SYSTEM_SEARCHER_PROMPT"
                                                 for target in node.targets)]
    if len(nodes) != 5:
        raise ValueError("pinned native generator definitions changed")
    body = "\n\n".join(ast.get_source_segment(source, node) for node in nodes) + "\n"
    def patch(old, new):
        nonlocal body
        if body.count(old) != 1:
            raise ValueError("pinned generator patch does not match: " + old)
        body = body.replace(old, new)
    patch("accept_rate = len(cleaned) / len(items)", "accept_rate = len(cleaned) / len(items) if items else 0.0")
    patch("    for it in items:\n", "    for raw_index, it in enumerate(items):\n")
    patch('"name": name.strip() + "_" + str(uuid.uuid4())[:6],',
          '"_raw_index": raw_index, "name": name.strip() + "_" + str(uuid.uuid4())[:6],')
    patch('        cleaned = [\n',
          '        if not all(isinstance(parsed_output[k], str) and parsed_output[k].strip() for k in ("name", "expression")):\n'
          '            return [], 0.0\n'
          '        cleaned = [\n')
    patch('"name": parsed_output["name"],', '"_raw_index": 0, "name": parsed_output["name"],')
    patch('"reason": parsed_output.get("reason", ""),',
          '"reason": parsed_output.get("reason", "") if isinstance(parsed_output.get("reason", ""), str) else "",')
    patch("expr_list = list(used_exprs)", "expr_list = ordered_expressions(used_exprs)")
    patch("        items, accept_rate = _normalize_llm_output(parsed_output)",
          "        raw_items(parsed_output)\n        items, accept_rate = _normalize_llm_output(parsed_output)\n        normalized_items(items)")
    patch("        for record in items:\n", "        for record in items:\n            visit_item(record)\n")
    patch("            if expr in used_exprs:\n", '            if expr in used_exprs:\n                item_status("duplicate")\n')
    patch('                quality_info["accepted_num"] += 1',
          '                item_status("accepted")\n                quality_info["accepted_num"] += 1')
    return body


class NativeGenerator:
    def __init__(self, protocol, runtime, client, gateway, collection, source_root):
        self.generator = Generator(protocol, runtime, client, gateway, collection)
        self.protocol, self.runtime, self.gateway = protocol, runtime, gateway
        if not gateway.mock and (client.temperature != protocol.temperature or
                                client.extra_body.get("text") != {"format": {"type": "json_object"}}):
            raise ValueError("native generation requires the frozen temperature and Responses JSON format")
        sources = {}
        for name, expected in SOURCES.items():
            raw = (Path(source_root) / name).read_bytes()
            if sha256(raw).hexdigest() != expected:
                raise ValueError("pinned native generator source changed: " + name)
            sources[name] = raw.decode("utf-8")
        body = generator_source(sources["agent/generator_qlib_search.py"])
        root = runtime.run_dir / "private/native_generator"
        manifest = {"sources": SOURCES, "patched_sha256": sha256(body.encode()).hexdigest(),
                    "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
        path = root / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("native generator changed across resume")
        atomic_json_write(path, manifest)
        (root / "generator.py").write_text(body, encoding="utf-8")
        self.code = compile(body, str(root / "generator.py"), "exec")
        self.prompts = {}
        exec(sources["agent/prompts_qlib_instruction.py"], self.prompts)
        self.choices = {event["event_key"]: event["payload"] for event in runtime.events()
                        if event["event_type"] == "native_generation_choice"}
        self.rng = random.Random(protocol.random_seed)
        for value in self.choices.values():
            if digest(value["value"]) != value["output_digest"]:
                raise EvaluationPaused("native generation choice integrity failure")
            if value["kind"] == "sample":
                state = value["value"]["after"]
                self.rng.setstate((state[0], tuple(state[1]), state[2]))

    def __call__(self, identity, instruction, model="deepseek-chat", N=10, max_try=5, avoid_repeat=False,
                 verbose=False, debug_mode=False, temperature=1., enable_reason=False, local=False, local_port=8000):
        if model != self.protocol.model or temperature != self.protocol.temperature or local:
            raise NativeStop(ValueError("native model parameters differ from the frozen provider contract"))
        if type(N) is not int or N < 1 or type(max_try) is not int or not 1 <= max_try <= 5:
            raise NativeStop(ValueError("native generation requires positive N and one to five attempts"))
        responses, occurrences, current_items, first_messages = [], [], {}, []
        state = {"attempt": -1, "choice": 0, "current": None}

        def choice(kind, inputs, draw):
            key = "native:generation-choice:" + digest([identity, state["choice"]])
            state["choice"] += 1
            def commit():
                saved = self.choices.get(key)
                input_digest = digest(inputs)
                if saved is None:
                    value = json.loads(json.dumps(draw()))
                    saved = {"generation": identity, "kind": kind, "input_digest": input_digest,
                             "value": value, "output_digest": digest(value)}
                    self.runtime.record("native_generation_choice", saved, event_key=key)
                    self.choices[key] = saved
                if saved["kind"] != kind or saved["input_digest"] != input_digest:
                    raise EvaluationPaused("native generation choice input changed")
                return json.loads(json.dumps(saved["value"]))
            return self.gateway.host.call(commit)

        def sample(population, count):
            def draw():
                before = self.rng.getstate()
                indices = self.rng.sample(range(len(population)), count)
                return {"before": before, "after": self.rng.getstate(), "indices": indices}
            saved = choice("sample", {"population": population, "count": count}, draw)
            return [population[index] for index in saved["indices"]]

        def call_llm(prompt, *, model, system_prompt, json_output, temperature, local, local_port, service_provider):
            state["attempt"] += 1
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            if not responses:
                first_messages.extend(messages)
            try:
                response = self.generator.request(["native", identity, state["attempt"]], messages)
            except (EvaluationPaused, BudgetExceededError) as exc:
                raise NativeStop(exc)
            except Exception as exc:
                raise NativeStop(EvaluationPaused("native model response requires inspection")) from exc
            responses.append(response)
            return response.text

        def raw_items(parsed):
            items = ([parsed] if isinstance(parsed, dict) and "name" in parsed and "expression" in parsed else
                     parsed.get("generated", []) if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else [])
            if not isinstance(items, list):
                items = []
            current_items.clear()
            for index, payload in enumerate(items):
                row = {"attempt": state["attempt"], "index": index, "payload": payload,
                       "status": "normalization_rejected", "normalized": False, "visited": False}
                occurrences.append(row)
                current_items[index] = row

        def normalized_items(items):
            for item in items:
                current_items[item["_raw_index"]].update(normalized=True, status="unprocessed")

        def visit_item(item):
            row = current_items[item["_raw_index"]]
            row.update(visited=True, status="unprocessed")
            state["current"] = row

        def item_status(status):
            state["current"]["status"] = status

        def check(expression):
            row = state["current"]
            candidate = self.generator.domain.admit(RawProposal({"expression": expression}, "native_generation"))
            if not isinstance(candidate, Candidate):
                row.update(status="static_rejected", error=candidate.reason)
                return {"success": False, "error": candidate.reason}
            position = ["native", identity, row["attempt"], row["index"]]
            try:
                result = self.gateway.evaluate(candidate, phase="check", position=position)
            except (EvaluationPaused, BudgetExceededError) as exc:
                raise NativeStop(exc)
            except Exception as exc:
                raise NativeStop(EvaluationPaused("native check response requires inspection")) from exc
            row["check_receipt"] = self.gateway.identity("check", position, candidate)
            valid = result["success"] and (self.protocol.filter_profile != "paper_filter_v1" or
                    result.get("nan_ratio", 1) <= .01 and result.get("elapsed_seconds", 31) <= 30)
            if not valid:
                row.update(status="dynamic_rejected", error=result.get("error", "dynamic quality threshold"))
            return {**result, "success": valid}

        namespace = {"json": json, "time": time, "Any": typing.Any, "Dict": typing.Dict,
            "Optional": typing.Optional, "Set": typing.Set,
            "QLIB_GENERATE_INSTRUCTION": self.prompts["QLIB_GENERATE_INSTRUCTION"],
            "ASSAY_GENERATE_INSTRUCTION": self.prompts["ASSAY_GENERATE_INSTRUCTION"],
            "is_assay": lambda: self.protocol.backend == "assay", "call_llm": call_llm,
            "check_factor_via_api": check, "raw_items": raw_items, "normalized_items": normalized_items,
            "visit_item": visit_item, "item_status": item_status,
            "ordered_expressions": lambda values: choice("expression_order", sorted(values), lambda: list(values)),
            "uuid": SimpleNamespace(uuid4=lambda: choice("uuid4", None, lambda: str(uuid.uuid4()))),
            "random": SimpleNamespace(sample=sample)}
        exec(self.code, namespace)
        result = namespace["call_qlib_search"](instruction, model, N, max_try, avoid_repeat, verbose,
            debug_mode, temperature, enable_reason, local, local_port)
        result = json.loads(json.dumps(result))
        generation = {"candidates": result["factors"], "occurrences": occurrences, "requested": N,
                      "attempt_count": len(responses), "complete": result["success"], "native_result": result,
                      "provider_settings": {"model": model, "wire_api": self.protocol.wire_api,
                          "reasoning_effort": self.protocol.reasoning_effort, "temperature_requested": temperature,
                          "temperature_effective": None, "temperature_status": "ignored_by_provider_in_thinking_mode",
                          "json_format": "json_object"},
                      "counts": {"raw_items": len(occurrences), "normalized_items": sum(r["normalized"] for r in occurrences),
                          "visited_items": sum(r["visited"] for r in occurrences),
                          "checked_items": sum("check_receipt" in r for r in occurrences),
                          "unique_accepted": len(result["factors"]),
                          "unprocessed_items": sum(r["status"] == "unprocessed" for r in occurrences)}}
        def commit():
            Receipts(self.runtime.run_dir / "generation").accept(["native", identity], generation)
            if result["success"]:
                ir = make_complete_design_ir(task_id="alphabench", domain="financial factor expressions",
                    task_description="Generate causal factors using only the native prompt and search observations.",
                    objectives=[{"name": self.protocol.objective, "direction": "maximize"}],
                    design_space_description=json.dumps(REGISTRY), observations=[], candidates=result["factors"],
                    request_description=json.dumps(first_messages), num_candidates=len(result["factors"]),
                    allows_new_parameters=False, reasoning_available=bool(enable_reason))
                self.generator.collection.accept(["native", identity], ir,
                    {"run": self.runtime.run_id, "protocol": self.protocol.identity})
        self.gateway.host.call(commit)
        return result
