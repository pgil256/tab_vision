# TabVision Architecture Brief

TabVision turns a single iPhone video of guitar playing into tablature. Audio
identifies when notes occur and what pitch they have; video constrains where
the fretting hand is on the neck; fusion chooses the playable string/fret path.

```mermaid
flowchart LR
    A["iPhone .mov"] --> B["Demux"]
    B --> C["Audio transcription"]
    B --> D["Video frames"]
    D --> E["Guitar / fretboard / hand tracking"]
    C --> F["Fusion"]
    E --> F
    F --> G["ASCII / GP5 / MusicXML / MIDI"]
    F --> H["Confidence map"]
```

Current demo status: scaffold only. Replace this with measured eval numbers
and screenshots after the integrated branch passes the Phase 8/9 gates.
