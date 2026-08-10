"""One shape for "I cannot answer", and one predicate that recognises it.

A consumer has to tell three states apart: there is no problem here, here is
the problem, and I could not look. The third was said eight different ways — a
bare ``[]``, an untyped ``error`` key beside real-looking zeros, prose on
stderr, a raised database error — and five of them at exit 0, which made "could
not look" indistinguishable from "looked, found nothing". That is the failure a
verify-first tool can least afford, because it reads as a clean bill of health.

The decision, written down in ``docs/notes/cannot-answer.md``:

* A payload that cannot answer carries the abstention marker and a non-empty
  ``reason``, at whatever level could not answer — skill row, CLI envelope, or
  web route body.
* "Ran cleanly but cannot answer" stays **exit 0**. The payload is the only
  signal, because an abstention is a correct and complete answer to the
  question that was asked. Exit 1 stays "the command did not finish" and exit 2
  stays "the command could not accept what it was given". A gate that was asked
  to *prove* something still exits non-zero when it cannot judge — that reports
  a failed assertion, not a failed command, which is why its payload must carry
  the marker too.

The shape is not new: ``skills.base.abstain`` has produced it at row level all
along, and ``skills.base`` remains its single definition site — this module
holds no copy of the marker string and builds nothing by hand, it delegates.
What is new is that the shape is canonical *above* the skill layer, and that
one predicate covers every level.

Both imports of ``skills.base`` are deferred to call time. Importing this
module therefore costs nothing, which matters: a contract nobody can reach
cheaply is one that gets re-implemented inline and drifts, and that is how
eight encodings happened in the first place.

Ownership should invert when ``skills/base.py`` is next opened: the marker
belongs here, with the skill layer importing it, rather than a general contract
reaching down into one of its consumers. It is left alone here so this change
stays a decision plus its test, and does not collide with the skill-layer work
that #345 has open in that file.
"""

from __future__ import annotations

__all__ = [
    "REASON_KEY",
    "cannot_answer",
    "cannot_answer_reason",
    "is_cannot_answer",
]

#: The sentence that says why. Required: a marker with no reason is the silence
#: this contract exists to remove, only typed.
REASON_KEY = "reason"


def cannot_answer(reason: str, **detail) -> dict:
    """Build the payload for a level that ran cleanly and cannot answer.

    The envelope-level companion to ``skills.base.abstain``, which returns the
    same object wrapped in the one-row list a skill must return; this returns
    the object itself, for an envelope or a route body. Extra keyword arguments
    are carried through, so a caller keeps the fields its existing consumers
    already read (see ``web.py``, which retains ``limitation``, ``error`` and
    ``cli`` beside the marker).

    ``reason`` must say something. Raises ``ValueError`` rather than asserting,
    because assertions are stripped under ``python -O`` and would let an empty
    reason reach a user precisely when the build is optimised.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a cannot-answer payload requires a non-empty reason")

    from .skills.base import abstain

    return abstain(reason, **detail)[0]


def is_cannot_answer(payload) -> bool:
    """True when ``payload`` says it could not answer, at any level.

    Apply it to the object you are about to read as an answer: ask the envelope
    before indexing ``payload["findings"]``, ask the row before indexing
    ``row["total_ms"]``.

    * ``dict``  — carries the marker itself.
    * ``list``  — non-empty and its first element carries the marker. That is
      what ``abstain()`` returns, and the rule is ``skills.base.is_abstention``
      exactly, called rather than restated.
    * anything else — never. ``None``, ``[]`` and ``""`` are not abstentions;
      an empty result means "ran, nothing to report", which is precisely the
      state the marker exists to distinguish from.
    """
    from .skills.base import is_abstention, is_abstention_row

    if isinstance(payload, dict):
        return is_abstention_row(payload)
    if isinstance(payload, list):
        return is_abstention(payload)
    return False


def cannot_answer_reason(payload) -> str:
    """The reason ``payload`` gives, or ``""`` when it is not an abstention.

    Exists so a renderer never re-implements the "is it a dict or the first row
    of a list" branch to reach one string.
    """
    if not is_cannot_answer(payload):
        return ""
    source = payload if isinstance(payload, dict) else payload[0]
    reason = source.get(REASON_KEY, "")
    return reason if isinstance(reason, str) else str(reason)
