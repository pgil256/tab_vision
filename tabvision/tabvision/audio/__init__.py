"""Audio module — see SPEC.md §3.3, §8.

Public entrypoint: ``transcribe_audio(wav, sr, session, backend) -> list[AudioEvent]``.

Backends:
- ``basicpitch`` — Phase 1 baseline (Spotify Basic Pitch).
- ``highres`` — Phase 2 SOTA swap (Riley/Edwards).
- ``tabcnn`` — Phase 2/5 tab-aware audio prior (trimplexx).
"""
