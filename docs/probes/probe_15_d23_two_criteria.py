"""D23: build the fixture that can tell the two refusal criteria apart.

The migration ledger's D23 records a semantic difference that is REGISTERED,
UNRULED and -- at the time it was written -- carried by no test in either
package:

* rheplicant's ``prior_sensitivity`` refused on the OBSERVED JACOBIAN's rank;
* bayesmith's refuses on the REST TERM's own curvature.

The line says a ruling needs a fixture that can separate them, and names the
shape: ``child ~ Normal(parent, s)`` with ``parent`` selected and ``child``
outside the selection. This probe builds it, and asks the two questions the
ruling actually turns on -- neither of which is "which criterion is better",
because that was settled when the module was written.

    (1) Is either direction of the difference REACHABLE from rheplicant's
        declaration layer? A semantic difference nothing can express is a
        different kind of item from one a user can hit.

    (2) Which criterion is running TODAY, on the facade rheplicant already
        ships? `prior_sensitivity` was switched in Wave A, so the answer is
        not necessarily the one rheplicant's own module docstring gives.

Exit code is 0 whenever the probe finished, never a verdict -- same rule as
`probe_11_d17_dual_run.py`, and for the same reason: a probe that turns
"the answer changed" into a non-zero exit invites the next reader to make it
green rather than to read it.

    cd /Users/zzhang/projects/bayesmith
    /Users/zzhang/projects/rheplicant/.venv/bin/python docs/probes/probe_15_d23_two_criteria.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist


def held_by_a_downstream_density():
    """``parent`` reaches the data ONLY through ``child``, whose density holds it.

    The observed node's Jacobian with respect to ``parent`` is identically
    zero at any fixed ``child`` -- so a rank test on it refuses. But ``child``'s
    own density ``Normal(parent, s)`` is part of the rest term, so the
    likelihood-only mode in ``parent`` is perfectly well defined: it is the
    value of ``child``.
    """
    from bayesmith import det, observe, sample, trace

    grid = jnp.linspace(0.5, 1.5, 12)

    def model():
        parent = sample("parent", lambda: dist.Normal(0.0, 3.0))
        # The whole point: child's DENSITY is parameterised by parent.
        child = sample("child", lambda p: dist.Normal(p, 0.5), parent)
        mu = det("mu", lambda c: c * grid, child)
        observe("d", lambda m: dist.Normal(m, 0.1).to_event(1), mu,
                obs=jnp.asarray(2.0 * grid))

    return trace(model)


def report(label: str, thunk) -> None:
    try:
        value = thunk()
    except Exception as error:  # noqa: BLE001 -- the verdict IS the exception
        print(f"  {label:<52} REFUSED  {type(error).__name__}")
        first = " ".join(str(error).split())
        print(f"  {'':<52}          {first[:150]}")
    else:
        print(f"  {label:<52} ACCEPTED {value}")


def question_one_reachability() -> None:
    """Can rheplicant's ParameterSpace declare a prior parameterised by a latent?"""
    print("\n[1] Is direction A expressible in rheplicant's declaration layer?")
    from rheplicant.inference.parameters import Latent

    print("      the shape D23 names is `child ~ Normal(parent, s)`, i.e. one")
    print("      latent's PRIOR parameterised by another latent's VALUE.")
    try:
        import inspect

        signature = inspect.signature(Latent.__init__ if hasattr(Latent, "__init__") else Latent)
        print(f"      Latent's declared fields: {list(signature.parameters)[1:]}")
    except (TypeError, ValueError):
        print(f"      Latent's declared fields: {getattr(Latent, '__annotations__', {})}")

    # A prior is a numpyro distribution built at DECLARATION time, so its
    # parameters are concrete arrays and there is no latent to reference yet.
    #
    # **Constructing one is NOT the test.** numpyro's Normal takes whatever it
    # is handed, so `dist.Normal(Latent(...), 0.5)` builds without complaint --
    # the first version of this probe read that as "direction A is reachable",
    # which is the same shape as a guard that cannot fail. What decides it is
    # whether the declaration survives being USED.
    parent = Latent("parent", init=0.0, prior=dist.Normal(0.0, 3.0))
    child = Latent("child", init=0.0, prior=dist.Normal(parent, 0.5))
    print(f"      Latent(prior=dist.Normal(<a Latent>, 0.5)) constructs: "
          f"{type(child.prior).__name__}  <-- numpyro took it without complaint")
    try:
        drawn = child.prior.sample(jax.random.key(0))
        print(f"      ... and sampling it gives: {drawn!r}")
        print("      -> direction A IS expressible (unexpected; read it twice)")
    except Exception as error:  # noqa: BLE001
        print(f"      ... but SAMPLING it fails: {type(error).__name__}: "
              f"{' '.join(str(error).split())[:120]}")
        print("      -> a prior parameterised by a Latent is not a declaration this")
        print("         layer can express; the constructor simply never checked.")


def question_two_which_criterion() -> None:
    """On the graph D23 names, which side refuses?"""
    from bayesmith.diagnose.sensitivity import prior_sensitivity

    print("\n[2] On the graph D23 names, what does each criterion say?")
    with jax.enable_x64(True):
        graph = held_by_a_downstream_density()
        values = {"parent": jnp.asarray(2.0), "child": jnp.asarray(2.0)}

        from bayesmith.diagnose.identifiability import identifiability

        rank_report = identifiability(graph, names=["parent"], at=values)
        print(f"      observed-Jacobian rank      : {rank_report.rank} of "
              f"{rank_report.n_par}  (nullity {rank_report.nullity})")
        print("      -> the JACOBIAN-RANK criterion would REFUSE"
              if rank_report.nullity
              else "      -> the JACOBIAN-RANK criterion would ACCEPT")

        report(
            "bayesmith prior_sensitivity(names=['parent'])",
            lambda: float(
                prior_sensitivity(graph, names=["parent"], at=values).shift_sigma[0]
            ),
        )


def question_three_the_other_direction() -> None:
    """Full observed rank, but a curvature the arithmetic cannot invert."""
    from bayesmith import det, observe, sample, trace
    from bayesmith.diagnose.identifiability import identifiability
    from bayesmith.diagnose.sensitivity import prior_sensitivity
    from bayesmith.exact.fisher import condition_ceiling

    print("\n[3] The OTHER direction: full Jacobian rank, digit-starved curvature")
    with jax.enable_x64(True):
        grid = jnp.linspace(0.0, 1.0, 16)
        # `grid * (1 + sep)` would be a scalar MULTIPLE of grid -- exactly
        # collinear at every separation, which is direction A's rank
        # deficiency wearing direction B's name. The second column has to
        # leave the first one's span by a controllable amount.
        for separation in (1e-1, 1e-3, 1e-5, 1e-7):
            columns = jnp.stack([grid, grid + separation * grid**2], axis=1)

            def model(columns=columns, grid=grid):
                a = sample("a", lambda: dist.Normal(0.0, 1.0))
                b = sample("b", lambda: dist.Normal(0.0, 1.0))
                mu = det("mu", lambda u, v: columns @ jnp.stack([u, v]), a, b)
                observe("d", lambda m: dist.Normal(m, 0.05).to_event(1), mu,
                        obs=jnp.asarray(grid))

            graph = trace(model)
            values = {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)}
            rank = identifiability(graph, names=["a", "b"], at=values)
            design = np.asarray(columns) / 0.05
            kappa = float(np.linalg.cond(design.T @ design))
            ceiling = float(condition_ceiling(jnp.zeros(1, dtype=jnp.float64).dtype))
            try:
                prior_sensitivity(graph, names=["a", "b"], at=values)
            except Exception as error:  # noqa: BLE001
                verdict = f"REFUSED ({type(error).__name__})"
            else:
                verdict = "ACCEPTED"
            print(f"      sep={separation:<8g} jacobian rank {rank.rank}/{rank.n_par}"
                  f"  kappa(H)={kappa:9.3e}  ceiling={ceiling:9.3e}  -> {verdict}")


def main() -> int:
    question_one_reachability()
    question_two_which_criterion()
    question_three_the_other_direction()
    print("\nExit 0 means the probe finished, not that the answers are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
