"""User-facing errors for the eval toolkit."""


class EvalObsError(Exception):
    """Base error. CLI commands print the message and exit."""


class PromptError(EvalObsError):
    """A prompt template is missing, duplicated, or cannot be rendered."""


class DatasetError(EvalObsError):
    """A test-case file is unreadable or does not match the case schema."""


class SuiteError(EvalObsError):
    """An eval suite file is unreadable or names an unknown scorer."""


class RubricError(EvalObsError):
    """A judge rubric is missing criteria or has an invalid scale."""


class ScorerError(EvalObsError):
    """A scorer cannot be built, for example because a regex is invalid."""


class LLMError(EvalObsError):
    """The model client failed or returned a payload this toolkit cannot use."""


class GateError(EvalObsError):
    """The score gate was configured with an invalid tolerance or baseline."""


class RunError(EvalObsError):
    """A saved eval run or baseline file cannot be read back."""
