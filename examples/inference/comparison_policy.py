"""Choose the comparisons presented in the inference notebook."""

COMPARISON_GALLERIES = {
    "proposals": ("Explicit proposal comparison", "显式提议对照"),
}
RETIRED_GALLERIES = {"jeffreys", "more-data", "mild-prior", "repeated-composed"}


def comparison_visible(kind, case):
    """Retired study outputs never become demo entries again."""
    return kind not in RETIRED_GALLERIES and case != "exponential_decay"
