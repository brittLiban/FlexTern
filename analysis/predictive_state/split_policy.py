"""Reserved-split policy (EXP-0013 Phase 1.1 closeout, hardening round). The layout split named "test" (ids 2000-2199) is reserved for locked
held-out evaluations. Ordinary exploratory / timing / diagnostic entrypoints must go through this module, so the default is TRAIN and the
only way to reach TEST is the explicit, auditable opt-in FLEXTERN_ALLOW_TEST_SPLIT=1 (used solely to regenerate frozen historical traces)."""
import os

OPT_IN = "FLEXTERN_ALLOW_TEST_SPLIT"
RESERVED = "test"


def test_split_allowed():
    return os.environ.get(OPT_IN) == "1"


def timing_split():
    """Split for ordinary timing / trace recording: TRAIN, or the reserved TEST range only with the opt-in (historical reproduction)."""
    return RESERVED if test_split_allowed() else "train"


def resolve(requested=None, default="train"):
    """Return `requested` (or `default`); asking for the reserved split without the opt-in raises."""
    split = requested or default
    if split == RESERVED and not test_split_allowed():
        raise PermissionError(f"split '{RESERVED}' is reserved; set {OPT_IN}=1 only to reproduce frozen historical results")
    return split
