# tabvision

Guitar tab transcription pipeline. Python CLI.

See `../SPEC.md` for the canonical project specification, and
`../docs/plans/2026-05-05-tabvision-spec-adoption-design.md` for the spec
adoption design.

This package is in **Phase 0** scaffold state — directory layout matches
SPEC.md §4, modules are stubs, contracts in `tabvision/types.py` match
SPEC.md §8.

## Install (dev)

```bash
cd tabvision
pip install -e '.[dev]'
```

## Test

```bash
pytest
```

A polished README with install / quickstart / cookbook ships in **Phase 9**
per SPEC.md §7 Phase 9.
