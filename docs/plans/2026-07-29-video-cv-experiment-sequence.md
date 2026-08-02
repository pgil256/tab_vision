# Video/Computer Vision Experiment Sequence

## Recommended sequence

1. Frozen RTMPose-versus-MediaPipe comparison—lowest-cost check.
2. UT contact-head prototype—highest likelihood of addressing the actual
   failure.
3. Measure not only overall video accuracy, but specifically
   `P(video correct | audio wrong)`. The earlier complementarity analysis
   shows that this is what must improve for fusion to help
   ([A14 report](/home/gilhooleyp/projects/tab_vision/docs/EVAL_REPORTS/a14_video_complementarity_2026-07-06.md)).
4. Use MMIP-derived pseudo-labels only if the small UT-trained contact model
   shows a promising source-disjoint signal.
5. Add CoTracker/SAM only if a frame-level audit identifies temporal dropout
   as a remaining bottleneck.

I would not prioritize another YOLO/fret-keypoint architecture, RF-DETR on the
same six-point labels, or training on generic hand datasets from scratch.
Those repeat supervision the project has already shown is insufficient.
