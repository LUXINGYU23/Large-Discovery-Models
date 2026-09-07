# Quick start

See [README.md](README.md) for the scientific adaptation and assets.

```bash
python scripts/run_ldm_tts.py config/reasyn/mock.yaml
python scripts/run_ldm_tts.py config/reasyn/mock_tdc.yaml
python -m pytest tasks/reasyn/tests -q
python tasks/reasyn/scripts/prepare_official_data.py --upstream-root ../ReaSyn-reasyn_v2
python scripts/check_task_dependencies.py config/reasyn/reconstruction_tiny.yaml
```

Mock outputs are synthetic. Real runs require frozen AR/EB checkpoints,
converted chemistry indices, the requested target data and LDM endpoint
settings. The task remains draft until a real seed/campaign is verified.
