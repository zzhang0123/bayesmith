"""Automatic affine discovery, separate from Gaussian solver eligibility.

Structural certificates and finite probes are independent evidence. Only a
structurally certified group with supported priors enters the exact compiler;
numerically plausible unknowns remain available to corrected proposal routes.
"""

import warnings
from dataclasses import dataclass

from bayesmith.diagnose.structure import conditional_affinity_certificate
from bayesmith.errors import NotGaussian, StructureError
from bayesmith.exact.block import _ancestors
from bayesmith.exact.linearity import DEFAULT_SCALES, check_linearity


@dataclass
class AffinityDiscovery:
    groups: tuple[tuple[str, ...], ...]
    evidence: dict[tuple[str, ...], dict]
    linearity: dict[tuple[str, ...], dict]
    reasons: dict[str, str]


def discover_affinity(graph, env, key, *, scales=DEFAULT_SCALES, rtol=None):
    """Find complete jointly certified groups, in graph declaration order."""
    # The classifier owns Gaussian capability checks and support-valid anchor
    # selection. Local imports avoid a dependency cycle with its entry point.
    from bayesmith.dispatch.classify import _at_points, _is_gaussian, block_at

    evidence, linearity, reasons = {}, {}, {}

    def proof(names):
        if names not in evidence:
            evidence[names] = conditional_affinity_certificate(graph, names, env)
        return evidence[names]

    def probe(names):
        record = proof(names)
        at = block_at(graph, names, env=env)
        points, note = _at_points(graph, names, env, key)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                measured = check_linearity(
                    graph,
                    names,
                    at,
                    at_points=points,
                    key=key,
                    scales=scales,
                    rtol=rtol,
                )
        except StructureError as error:
            record.update(numerical_status="counterexample", numerical_detail=str(error))
            return False
        except (NotGaussian, ValueError, TypeError, NotImplementedError) as error:
            record.update(numerical_status="unknown", numerical_detail=str(error))
            return False
        linearity[names] = measured
        warning_note = "; ".join(str(item.message) for item in caught)
        record.update(
            numerical_status="passed",
            numerical_note="; ".join(item for item in (note, warning_note) if item),
        )
        return True

    candidates = []
    for name in graph.latents:
        names = (name,)
        record = proof(names)
        record["numerical_status"] = "not_checked"
        # All prediction structures are inspected, including non-Gaussian
        # priors. Eligibility is a separate, downstream decision.
        ok, why = _is_gaussian(graph, name, env)
        covered_by_evidence = any(name in term.over for term in graph.evidence_terms)
        if covered_by_evidence:
            reasons[name] = (
                f"{name!r} is covered by a graph-level evidence term and must "
                "remain in NUTS so its density is evaluated"
            )
        elif any(name in _ancestors(graph, other) for other in graph.latents if other != name):
            reasons[name] = f"{name!r} is an ancestor of another latent's distribution"
        elif not ok:
            reasons[name] = why
        elif not record["gaussian_priors"]:
            reasons[name] = "conditional Gaussian prior not structurally certified"
        else:
            passed = probe(names)
            if record["certified"] and passed:
                candidates.append(name)
                continue
            reasons[name] = record.get("numerical_detail", record["reason"])
            if passed:
                reasons[name] += "; numerical probes passed, global affinity unproved"

    groups = []
    for name in candidates:
        for index, group in enumerate(groups):
            combined = group + (name,)
            record = proof(combined)
            if record["certified"] and record["gaussian_priors"]:
                groups[index] = combined
                break
        else:
            groups.append((name,))

    accepted = []
    for group in groups:
        # The COMPLETE group is checked. Pairwise finite observations are not
        # promoted into a joint certificate, even when every outside anchor is zero.
        if group in linearity or probe(group):
            accepted.append(group)
        else:
            for name in group:
                reasons[name] = evidence[group].get("numerical_detail", "joint probe unresolved")
    return AffinityDiscovery(tuple(accepted), evidence, linearity, reasons)
