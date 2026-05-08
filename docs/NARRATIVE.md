# TabVision Narrative

TabVision is a portfolio-oriented transcription project for solo guitar videos.
The core problem is that pitch alone does not determine tablature: the same
note can be played on several strings. The v1 architecture keeps audio,
vision, fusion, and rendering as separate modules so each source of evidence
can improve without entangling the rest of the pipeline.

What works now:

- v1 package/CLI scaffold with strict dataclass contracts.
- Audio, video, and fusion modules under active phased integration.
- Confidence-aware ASCII output and Phase 6 renderer scaffolding for GP5,
  MusicXML, and MIDI.
- HTML diagnose report scaffold for inspecting one clip.
- License-clean default check scaffold for Phase 9.

What remains before a public v1 release:

- Automated eval evidence from deterministic smoke fixtures and public datasets.
- Final demo recordings and side-by-side output examples from existing or
  generated assets.
- Final license gate after all default backends are selected.
- Release tag only after coordinator integration and user sign-off.

Full hand-labeled evaluation on newly recorded user clips is intentionally
future validation, not a v1 prerequisite. The release story should be explicit:
manual annotation was removed from the critical path so the project can ship
with reproducible automated evidence rather than private/manual gates.
