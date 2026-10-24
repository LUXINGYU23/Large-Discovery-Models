"""Host-only, serialized oracle transport with durable per-campaign receipts."""
from __future__ import annotations
import hashlib
import json
import math
import socket
import socketserver
import time
from pathlib import Path
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import atomic_json_write
from tasks.nucleobench.core.constants import UPSTREAM_COMMIT
from tasks.nucleobench.core.digests import file_digest

MAX_MESSAGE = 16 * 1024 * 1024


def oracle_identity(prepared):
    value = {'source_commit': UPSTREAM_COMMIT, 'case': prepared.context.case.case_id,
             'model_sha256': file_digest(prepared.model_artifact),
             'model_init_args': prepared.model_init_args, 'precision': 'float32_tf32_disabled'}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def request_identity(namespace, identity, sequences):
    hashes = [hashlib.sha256(s.encode('ascii')).hexdigest() for s in sequences]
    return hashlib.sha256(json.dumps([namespace, identity, hashes]).encode()).hexdigest()


class QueuedOracle:
    def __init__(self, path: Path, namespace: str, identity: str, *, timeout: float = 3600):
        self.path, self.namespace, self.identity, self.timeout = path, namespace, identity, timeout

    def __call__(self, sequences):
        sequences = list(sequences)
        request_id = request_identity(self.namespace, self.identity, sequences)
        payload = {'namespace': self.namespace, 'identity': self.identity,
                   'request_id': request_id, 'sequences': sequences, 'queued_at': time.time()}
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(self.timeout)
                conn.connect(str(self.path))
                conn.sendall(json.dumps(payload).encode() + b'\n')
                with conn.makefile('rb') as stream:
                    response = json.loads(stream.readline(MAX_MESSAGE))
            if response.get('request_id') != request_id or response.get('identity') != self.identity:
                raise ValueError('Oracle response identity mismatch')
            if response.get('status') != 'completed':
                raise ValueError(response.get('error', 'Oracle result unresolved'))
            energies = response['energies']
            if len(energies) != len(sequences) or not all(math.isfinite(float(x)) for x in energies):
                raise ValueError('Invalid oracle energies')
            return energies
        except (OSError, ValueError, KeyError) as exc:
            raise EvaluationPaused(f'Queued oracle requires reconciliation: {exc}') from exc


class OracleQueue(socketserver.UnixStreamServer):
    """One worker and one model. The socket accept backlog is the FIFO queue."""
    request_queue_size = 128

    def __init__(self, path, model, identity, receipts, *, max_batch_size=4):
        self.model, self.identity = model, identity
        self.receipts = Path(receipts)
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.max_batch_size = max_batch_size
        super().__init__(str(path), _Handler)
        Path(path).chmod(0o600)

    def score(self, request):
        sequences = request['sequences']
        if request['identity'] != self.identity:
            raise ValueError('Oracle scientific identity mismatch')
        if not 1 <= len(sequences) <= self.max_batch_size or any(not isinstance(s, str) for s in sequences):
            raise ValueError('Invalid inference minibatch')
        key = request_identity(request['namespace'], self.identity, sequences)
        if request['request_id'] != key:
            raise ValueError('Oracle request digest mismatch')
        path = self.receipts / (key + '.json')
        if path.exists():
            prior = json.loads(path.read_text())
            if prior['status'] != 'completed':
                raise ValueError('Previous dispatch has unknown outcome; refusing to repeat inference')
            return {**prior, 'replayed': True}
        record = {'request_id': key, 'identity': self.identity, 'namespace': request['namespace'],
                  'sequence_count': len(sequences), 'status': 'dispatched',
                  'queued_at': request['queued_at'], 'started_at': time.time()}
        atomic_json_write(path, record)
        energies = [float(x) for x in self.model(sequences)]
        if len(energies) != len(sequences) or not all(math.isfinite(x) for x in energies):
            raise ValueError('Official oracle returned invalid energies')
        record.update(status='completed', energies=energies, finished_at=time.time())
        atomic_json_write(path, record)
        return {**record, 'replayed': False}


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(120)
        request = {}
        try:
            raw = self.rfile.readline(MAX_MESSAGE + 1)
            if len(raw) > MAX_MESSAGE or not raw.endswith(b'\n'):
                raise ValueError('Invalid oracle request framing')
            request = json.loads(raw)
            result = self.server.score(request)
        except Exception as exc:
            result = {'request_id': request.get('request_id'), 'identity': self.server.identity,
                      'status': 'unresolved', 'error': str(exc)}
        try:
            self.wfile.write(json.dumps(result).encode() + b'\n')
        except (BrokenPipeError, ConnectionResetError):
            pass  # Completed receipts remain available after a disconnected client.
