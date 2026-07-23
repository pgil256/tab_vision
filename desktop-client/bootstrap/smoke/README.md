# First-run smoke fixture

`test_a440_5s.mp4` is a five-second loop of TabVision's checked-in synthetic
440 Hz fixture. It contains no private/user recording. The pinned desktop CLI
runs its real high-resolution audio backend against this input with preflight
and video disabled, then compares the ASCII output byte-for-byte with
`expected.tab` before first-run setup is considered healthy.

The fixture is exactly 5.000 seconds, 122,080 bytes, with SHA-256
`e8c1e86d96cdba84e50b6b1202fbaffc33abc2dceedb482fe040faa477e3953c`.
The expected output is 222 bytes, with SHA-256
`6f31038967923c6525e28ece6bad766be44acbe6f709a18cbd65c754ad259ce3`.
