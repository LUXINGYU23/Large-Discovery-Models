"""Task discovery and registration interface."""

from ldm_tts.registration.async_manifest import (
    AsyncHookDeclaration, AsyncManifest, AsyncManifestError, HookLoadError,
    load_hook, parse_async_manifest,
)

from ldm_tts.registration.registry import (
    TaskDefinition,
    TaskRegistrationError,
    discover_task_definitions,
    get_task_definition,
)
from ldm_tts.registration.qualification import (
    QUALIFICATION_STAGES,
    QualificationEvidence,
    QualificationEvidenceError,
    load_qualification_evidence,
)

__all__ = [
    "AsyncHookDeclaration", "AsyncManifest", "AsyncManifestError", "HookLoadError",
    "load_hook", "parse_async_manifest",
    "TaskDefinition",
    "TaskRegistrationError",
    "QUALIFICATION_STAGES",
    "QualificationEvidence",
    "QualificationEvidenceError",
    "discover_task_definitions",
    "get_task_definition",
    "load_qualification_evidence",
]
