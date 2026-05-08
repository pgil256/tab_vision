# GuitarSet Audio Eval (highres)

Split: **validation**
Tracks: **3**
Gold notes: **288**
Audio events: **296**

## Aggregate

| Metric | Mean F1 | Micro P | Micro R | Micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Onset | 0.961 | 0.956 | 0.983 | 0.969 |
| Pitch | 0.948 | 0.943 | 0.969 | 0.955 |
| Tab | 0.315 | 0.331 | 0.340 | 0.336 |

## Per Track

| Track | Gold | Audio | Decoded | Onset F1 | Pitch F1 | Tab F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `05_BN1-129-Eb_comp` | 148 | 150 | 150 | 0.980 | 0.973 | 0.221 |
| `05_BN1-129-Eb_solo` | 44 | 46 | 46 | 0.933 | 0.933 | 0.111 |
| `05_BN1-147-Gb_comp` | 96 | 100 | 100 | 0.969 | 0.939 | 0.612 |
