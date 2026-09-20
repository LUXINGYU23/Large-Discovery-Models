# Versioned resources

- `source_manifest.json`: hash identity of the official source archive; no fabricated Git revision.
- `mock_fixture.json`: synthetic public question, private mock label and deterministic responses, used only by host-side tests.
- `harness/`: task roles, selected skills, structured public tools and the guest image recipe, including the delivered geometry package. This is the only task resource tree mounted into agents.
- `qualification_evidence.json`: formal qualification remains draft/scaffolded until live research and evaluation traces are verified.
- `local_verification.json`: historical pre-review evidence; its date and artifact hashes describe that historical snapshot.
- `review_verification.json`: validation of the review repairs and remaining real-runtime requirements.

Released datasets and run outputs belong outside versioned resource trees. Private labels and judge artifacts must never enter agent mounts or prompts.
