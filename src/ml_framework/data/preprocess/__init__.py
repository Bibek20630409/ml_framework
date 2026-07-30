"""Preprocessors: every piece of fitted transform state, and nothing else.

Import the concrete classes from their own modules (``preprocess.tabular``,
``preprocess.image``) rather than eagerly here — ``image`` reaches for torchvision
inside its methods, and keeping the package ``__init__`` light preserves the rule
that a bare install can import the whole data layer.
"""

from .base import (
    MANIFEST_NAME,
    BasePreprocessor,
    IdentityPreprocessor,
    PreprocessorError,
    load_preprocessor,
)

__all__ = [
    "BasePreprocessor",
    "IdentityPreprocessor",
    "PreprocessorError",
    "load_preprocessor",
    "MANIFEST_NAME",
]
