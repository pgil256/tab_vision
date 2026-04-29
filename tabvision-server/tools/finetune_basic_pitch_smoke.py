"""5-epoch overfit smoke test for the basic_pitch fine-tune loop.

Phase 1 Week 1 deliverable per docs/plans/2026-04-24-audio-backbone-finetune-design.md §7
("Friday gate"): confirm the training loop runs and the loss decreases on a
5-clip overfit. If this can't be made to run cleanly, the plan's bailout is
to pivot to a frozen-encoder + new-head approach (HuBERT / MusicFM).

Inputs: TFRecords produced by `tools/build_guitarset_tfrecords.py` (under
`tools/outputs/tfrecords/guitarset/splits/{train,validation}/`).

Outputs: Keras checkpoints + tensorboard under `tools/outputs/finetune_smoke/`.

This script bypasses `basic_pitch.train.console_entry_point` because that
entrypoint has a bug (positional arg order mismatch and a missing
`dont_sonify` attribute). We call `train.main` directly with documented
positional order.

Usage:
    python -m tools.finetune_basic_pitch_smoke
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tools', 'outputs', 'tfrecords',
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tools', 'outputs', 'finetune_smoke',
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default=DEFAULT_SOURCE,
                    help='base directory containing {dataset}/splits/{train,validation}/*.tfrecord')
    ap.add_argument('--output', default=DEFAULT_OUTPUT,
                    help='output directory for tensorboard + checkpoints')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--steps-per-epoch', type=int, default=5)
    ap.add_argument('--validation-steps', type=int, default=1)
    ap.add_argument('--shuffle-size', type=int, default=8)
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--no-contours', action='store_true', default=False)
    ap.add_argument('--no-sonify', action='store_true', default=True,
                    help='skip sonification in visualize callback (default on; faster)')
    ap.add_argument('--no-visualize', action='store_true', default=False,
                    help='strip the VisualizeCallback entirely (skip if it breaks)')
    args = ap.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)

    # Heavy imports after CLI parse so --help is fast.
    from basic_pitch import train as bp_train

    if args.no_visualize:
        # Monkey-patch VisualizeCallback to a no-op so the loop doesn't choke
        # on missing nnAudio / sonify dependencies. Smoke test only cares
        # about the actual fit loop running.
        import tensorflow as tf

        class _NoopVis(tf.keras.callbacks.Callback):
            def __init__(self, *a, **k):
                super().__init__()

        bp_train.VisualizeCallback = _NoopVis  # type: ignore[attr-defined]

    print('--- smoke params ---', file=sys.stderr)
    print(f'  source: {args.source}', file=sys.stderr)
    print(f'  output: {args.output}', file=sys.stderr)
    print(f'  epochs: {args.epochs}, batch_size: {args.batch_size}, '
          f'steps_per_epoch: {args.steps_per_epoch}, '
          f'validation_steps: {args.validation_steps}', file=sys.stderr)

    bp_train.main(
        source=args.source,
        output=args.output,
        batch_size=args.batch_size,
        shuffle_size=args.shuffle_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        size_evaluation_callback_datasets=2,
        datasets_to_use=['guitarset'],
        dataset_sampling_frequency=np.array([1.0]),
        no_sonify=args.no_sonify,
        no_contours=args.no_contours,
        weighted_onset_loss=False,
        positive_onset_weight=0.5,
    )
    print('done.', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
