"""Frozen, code-only representations. Candidate execution is never part of encoding."""
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ldm_tts.contracts import SurrogateSpaceSpec
from ldm_tts.optimization.records import SurrogateVector
from ldm_tts.engine.run_store import atomic_json_write
from .candidate import validate_source

AST_DIMENSION = 256
QWEN_DIMENSION = 1024
MODEL_REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
NORMALIZATION = 'python-ast-unparse-v1'
FEATURE_VERSION = 'copper-normalized-ast256-v2'
DIMENSION = AST_DIMENSION


def normalized_code(source):
    # Keep names, constants, docstrings, and statement order. Do not execute/import.
    return ast.unparse(ast.parse(source.decode('utf-8'))) + '\n'


def unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError('invalid code representation')
    return value / norm


def ast_features(code):
    vector = np.zeros(AST_DIMENSION, dtype=np.float64)
    def add(feature):
        digest = hashlib.sha256(feature.encode()).digest()
        vector[int.from_bytes(digest[:8], 'big') % AST_DIMENSION] += 1 if digest[8] & 1 else -1
    for node in ast.walk(ast.parse(code)):
        kind = type(node).__name__
        add(kind)
        for name, value in ast.iter_fields(node):
            if not isinstance(value, (ast.AST, list)):
                add(f'{kind}.{name}={value!r}')
    return unit(vector)


def encoding_spec(mode='ast'):
    dimensions = {'ast': 256, 'qwen': 1024, 'fusion': 1280}
    if mode not in dimensions:
        raise ValueError('unknown encoding')
    version = FEATURE_VERSION if mode == 'ast' else f'copper-{NORMALIZATION}-{mode}-{MODEL_REVISION}-v1'
    return SurrogateSpaceSpec(kind='vector', representation=f'Normalized code; {mode}; block L2; fusion sqrt(0.5) weights',
                              dimension_policy='fixed', dimension=dimensions[mode], version=version)


class QwenEmbedding:
    """Local frozen Sentence Transformers encoder; fail instead of truncating code."""
    def __init__(self, path, device):
        from sentence_transformers import SentenceTransformer
        path = Path(path).resolve()
        provenance = json.loads((path / 'provenance.json').read_text())
        if provenance['model_id'] != 'Qwen/Qwen3-Embedding-0.6B' or provenance['revision'] != MODEL_REVISION:
            raise ValueError('embedding model revision mismatch')
        for name, expected in provenance['file_sha256'].items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
                raise ValueError('embedding model checksum mismatch: ' + name)
        import torch
        torch.set_num_threads(1)
        self.model = SentenceTransformer(str(path), device=device, local_files_only=True,
                                         model_kwargs={'dtype': 'float32'})
        self.model.max_seq_length = 32768
        self.model.eval()

    def encode(self, code):
        tokens = self.model.tokenizer(code, truncation=False, add_special_tokens=True)['input_ids']
        if len(tokens) > self.model.max_seq_length:
            raise ValueError(f'normalized code exceeds embedding context: {len(tokens)} tokens; no truncation')
        vector = self.model.encode([code], normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False, batch_size=1, prompt='')[0]
        if vector.shape != (QWEN_DIMENSION,):
            raise ValueError('unexpected Qwen embedding dimension')
        return unit(vector), len(tokens)


class CodeEncoder:
    def __init__(self, root, mode='ast', model_path=None, device='cuda:0'):
        self.root, self.mode = Path(root).resolve(), mode
        self.spec = encoding_spec(mode)
        self.model_path, self.device, self.model = model_path, device, None

    def describe(self):
        return self.spec

    def encode(self, candidate):
        started = time.monotonic()
        path = self.root / candidate.payload['source_path']
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError('source escaped sealed candidate root')
        source = path.read_bytes()
        if validate_source(source) != candidate.canonical_key:
            raise ValueError('source digest changed before encoding')
        code = normalized_code(source)
        code_sha = hashlib.sha256(code.encode()).hexdigest()
        target = self.root / 'encodings' / f'{candidate.candidate_id}.json'
        if target.exists():
            record = json.loads(target.read_text())
            if record['version'] != self.spec.version or record['normalized_sha256'] != code_sha:
                raise ValueError('encoding cache identity mismatch')
            vector = np.asarray(record['values'], dtype=np.float64)
            if vector.shape != (self.spec.dimension,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1.):
                raise ValueError('invalid cached embedding')
        else:
            a = ast_features(code)
            e, tokens, load_seconds = None, None, 0.
            if self.mode != 'ast':
                if self.model is None:
                    before = time.monotonic()
                    self.model = QwenEmbedding(self.model_path, self.device)
                    load_seconds = time.monotonic() - before
                e, tokens = self.model.encode(code)
            vector = a if self.mode == 'ast' else e if self.mode == 'qwen' else np.concatenate((a, e)) / np.sqrt(2.)
            record = dict(source_sha256=candidate.canonical_key, normalized_sha256=code_sha,
                          normalization=NORMALIZATION, normalized_code=code, version=self.spec.version,
                          mode=self.mode, values=vector.tolist(), ast_values=a.tolist(),
                          qwen_values=e.tolist() if e is not None else None, tokens=tokens,
                          elapsed_seconds=time.monotonic()-started, model_load_seconds=load_seconds,
                          model_revision=MODEL_REVISION if e is not None else None)
            atomic_json_write(target, record)
        return SurrogateVector(tuple(float(v) for v in vector), self.spec.version, source_id=candidate.candidate_id)
