"""Keep the example demos off each other's back, because they do not fit.

WHY THIS FILE EXISTS, measured. The nightly Linux suite failed nine nights
running, 2026-09-12 through 2026-09-20, and not one of those runs reached a
summary line: every one was killed mid-suite with exit 143 and the annotation
"The hosted runner lost communication with the server ... starves it for
CPU/Memory". Progress decelerated rather than stopped -- in run 34682061310
the source job went 48% -> 69% in ELEVEN SECONDS and 69% -> 74% in FOURTEEN
MINUTES -- which is swap, not a slow test. A slow test is slow at a constant
rate; only a resource leak accelerates.

The cause is that these demos are run in a CHILD INTERPRETER, one per test,
and one of them is enormous. Peak RSS of a single child, measured on this
checkout on 2026-09-20 with `resource.getrusage(RUSAGE_CHILDREN)`:

    composed_process            4765 MB      <-- 3840 observations, x64 NUTS
    validate_sampling --smoke    928 MB
    hierarchical                 817 MB
    multiplicative_noise         768 MB
    linear_gaussian              585 MB
    hierarchy --quick            520 MB
    three_routes --quick         517 MB

Importing bayesmith, jax and numpyro costs a further 151 MB in every worker.

Under xdist's default `--dist load` there is nothing stopping four workers
from each reaching an example test at once, and four children alone come to
4765 + 928 + 817 + 768 = 7278 MB before the workers that spawned them. The
runner does not have that. This is arithmetic, not a race that got unlucky,
which is why it reproduced every single night.

THE DEMOS ARE NOT THE DEFECT AND MUST NOT BE SHRUNK. Their budgets are
registered settings and tests/test_inference_examples.py asserts on what they
recover -- every generating value inside its marginal 99% interval, posterior
SD under half the prior SD. Cutting draws to fit the runner would buy a green
run by weakening the claim, which is the move this repository forbids in
CLAUDE.md ("do not widen a tolerance to get to green"). The budget is correct
and the SCHEDULING was wrong.

So the children are serialised instead: every test in these modules is put in
one xdist group, and CI runs `--dist loadgroup`, which sends a group to a
single worker. At most one child then exists at a time. The cost is bounded
and small -- the whole group is about two minutes of a forty-minute suite.

The group is INERT unless `--dist loadgroup` is passed, so a plain `-n 4` here
behaves exactly as it did before; on a development machine with the memory to
spare that is the faster arrangement and nothing is lost.

test_example_serialization.py asserts this list still names every module that
launches an example in a child interpreter, so adding a module and forgetting
this file turns the suite red rather than the nightly.
"""

from __future__ import annotations

import pytest

#: Test modules that run an example in a child interpreter. Serialised onto one
#: xdist worker under `--dist loadgroup`; see this module's docstring.
SERIAL_EXAMPLE_MODULES = frozenset(
    {
        "test_inference_examples",
        "test_examples",
        # NOTE on this branch: main also carries test_campbell_scaling, a
        # plotting entry point that satisfies the same property. It does not
        # exist on this base, and the guard below is an equality, so naming it
        # here would fail rather than quietly over-serialise. That is the guard
        # working: the list is checked against the tests that are actually
        # present, not against a memory of another branch.
    }
)

#: One name, so every module above lands on the same worker.
SERIAL_EXAMPLE_GROUP = "example-subprocess"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Put every example-running test in one xdist group.

    Marking at collection time rather than in each file keeps the reason in one
    place, and keeps the test modules readable as tests rather than as
    scheduling configuration.

    `tryfirst` IS LOAD-BEARING AND WAS MEASURED, not chosen defensively. xdist
    reads `xdist_group` in a `pytest_collection_modifyitems` of its own, in the
    worker, and rewrites the item's nodeid to carry the group -- that rewritten
    nodeid is the only thing the controller's scheduler ever sees. A marker
    added after that hook has already run is therefore real, visible to
    `iter_markers`, and completely ignored. Measured on 2026-09-20 without
    `tryfirst`, `--dist loadgroup -n 2` over the three modules still split them
    gw0:5 / gw1:20; with it they land on one worker. The plain version looked
    right, collected right, and scheduled exactly as if the file did not exist.
    """
    for item in items:
        if item.module.__name__.rpartition(".")[2] in SERIAL_EXAMPLE_MODULES:
            item.add_marker(pytest.mark.xdist_group(SERIAL_EXAMPLE_GROUP))
