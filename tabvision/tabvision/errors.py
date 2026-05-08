"""Exception hierarchy — see SPEC.md §5.4."""


class TabVisionError(Exception):
    """Base for all TabVision errors."""


class InvalidInputError(TabVisionError):
    """Bad input from the caller (file missing, wrong format, etc.)."""


class BackendError(TabVisionError):
    """A backend (audio model, vision tracker) failed."""


class FusionError(TabVisionError):
    """The fusion stage failed to produce a coherent decoding."""
