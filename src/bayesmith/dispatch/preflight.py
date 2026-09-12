"""Bounded, explicit pre-sampling findings; diagnostics never choose a prior."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith.artifacts import (
    AnalysisFinding,
    InitializationPolicy,
    NamedArray,
    ScopeKind,
    ScopeRef,
)
from bayesmith.diagnose.identifiability import DEFAULT_RANK_RTOL, identifiability
from bayesmith.diagnose.local import (
    check_differentiable,
    flat_view,
    local_block,
    refuse_ambient_float32,
    refuse_single_precision,
    resolve_names,
    unflatten,
)
from bayesmith.diagnose.structure import (
    derivative_semantics_evidence,
    gaussian_flatness_certificate,
)
from bayesmith.dispatch.initialization import complete_initial_values
from bayesmith.dispatch.pareto_information import pareto_information, pareto_parameters
from bayesmith.errors import BayesmithError, GraphError
from bayesmith.exact.block import _ancestors
from bayesmith.exact.fisher import (
    _log_spectrum_curvature,
    _spans,
    _weighted_design,
    dense_operator,
)
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.gls import precision_from_graph
from bayesmith.graph.evaluate import apply_probabilistic, evaluate

_UNAVAILABLE = (
    BayesmithError,
    ValueError,
    TypeError,
    NotImplementedError,
    np.linalg.LinAlgError,
)


def _finding(code, status, scope="model", **measurements):
    return AnalysisFinding(
        code,
        status,
        ScopeRef(ScopeKind.MODEL if scope == "model" else ScopeKind.BLOCK, scope),
        tuple(measurements.items()),
        ("pre_sampling_diagnostic",),
    )


def _certificate_value(value):
    """Freeze the helper's evidence dict into the report's canonical values."""
    if isinstance(value, dict):
        return tuple((key, _certificate_value(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_certificate_value(item) for item in value)
    return value


def _bernoulli_base(distribution):
    """Only canonical wrappers with independent, unchanged density semantics."""
    while type(distribution) in (dist.Independent, dist.ExpandedDistribution):
        distribution = distribution.base_dist
    return distribution


def _observation_shape(graph, node, distribution):
    # NumPyro's single graph plate occupies the last BATCH axis, before
    # Independent event dimensions. A scalar obs does not remove replication.
    batch_shape = distribution.batch_shape
    if node.plate:
        batch_shape = jnp.broadcast_shapes(
            batch_shape, (graph.plate_size(node.plate[0]),)
        )
    return jnp.broadcast_shapes(
        batch_shape + distribution.event_shape, jnp.shape(node.observed)
    )


def _check_bernoulli_probs(graph, env):
    for name in graph.observed:
        base = _bernoulli_base(apply_probabilistic(graph, graph.node(name), env))
        if type(base) is not dist.BernoulliProbs:
            continue
        probability = np.asarray(base.probs)
        bounds = np.finfo(jnp.result_type(base.probs, float))
        if (
            not np.all(np.isfinite(probability))
            or np.any(probability <= bounds.tiny)
            or np.any(probability >= 1 - bounds.eps)
        ):
            raise GraphError(
                f"BernoulliProbs on {name!r} has boundary/invalid probabilities "
                "or reaches NumPyro's density clipping interval endpoints; "
                "regular information requires finfo.tiny < probs < 1-finfo.eps. "
                "Clipped regions and their derivative kinks are unsupported."
            )


def _bernoulli_logits(graph, name, env):
    node = graph.node(name)
    distribution = apply_probabilistic(graph, node, env)
    base = _bernoulli_base(distribution)
    if type(base) not in (dist.BernoulliLogits, dist.BernoulliProbs):
        raise GraphError(
            f"Bernoulli information does not support observed node {name!r} "
            f"with {type(base).__name__}; every observation must be a canonical "
            "Bernoulli, optionally Independent/Expanded. Mixed families, "
            "distribution masks and custom subclasses are unsupported."
        )
    if node.observed_mask is not None and distribution.event_shape:
        raise GraphError(
            f"Bernoulli information does not support event-reduced observation "
            f"masks on {name!r}; use unreduced Bernoulli batch observations."
        )
    # NumPyro clips Probs.log_prob to [finfo.tiny, 1-finfo.eps]. The concrete
    # guard excludes those boundaries/regions; only within that open interval
    # do these unclipped log odds describe the density actually sampled.
    logits = (
        base.logits
        if type(base) is dist.BernoulliLogits
        else jnp.log(base.probs) - jnp.log1p(-base.probs)
    )
    shape = _observation_shape(graph, node, distribution)
    return jnp.broadcast_to(logits, shape)


def _bernoulli_information(graph, names, values):
    """Expected score outer product: J_logit.T diag(p (1-p)) J_logit."""
    check_differentiable(graph, names, values)
    if any(_ancestors(graph, n).intersection(graph.observed) for n in graph.observed):
        raise GraphError(
            "Bernoulli information does not support observation-dependent logits; "
            "the expectation over upstream outcomes has not been performed."
        )
    env = evaluate(graph, values)
    _check_bernoulli_probs(graph, env)
    point, shapes, spans = flat_view(values, names)

    def logits_at(vector):
        env = evaluate(graph, {**values, **unflatten(vector, names, shapes, spans)})
        return jnp.concatenate(
            [jnp.ravel(_bernoulli_logits(graph, name, env)) for name in graph.observed]
        )

    logits = logits_at(point)
    refuse_single_precision(logits, doing="automatic Bernoulli likelihood geometry")
    if not np.all(np.isfinite(logits)):
        raise GraphError(
            "Bernoulli information requires finite logits and probabilities "
            "strictly between zero and one; boundary/invalid probabilities "
            "do not define this regular Fisher calculation."
        )
    jacobian = np.asarray(jax.jacfwd(logits_at)(point))
    refuse_single_precision(jacobian, doing="automatic Bernoulli likelihood geometry")
    # sigmoid(-eta) keeps 1-p accurate even when sigmoid(eta) rounds to one.
    weights = np.asarray(jax.nn.sigmoid(logits) * jax.nn.sigmoid(-logits)).copy()
    offset = 0
    for name in graph.observed:
        shape = _bernoulli_logits(graph, name, env).shape
        count = int(np.prod(shape))
        mask = graph.node(name).observed_mask
        if mask is not None:
            weights[offset : offset + count] *= np.broadcast_to(mask, shape).ravel()
        offset += count
    if not np.all(np.isfinite(jacobian)):
        raise GraphError(
            "Bernoulli logit Jacobian is nonfinite at the diagnostic point"
        )
    matrix = jacobian.T @ (weights[:, None] * jacobian)
    return np.asarray((matrix + matrix.T) / 2)


def likelihood_information(graph, names, values):
    """Expected Fisher of direct Gaussian, Bernoulli or fixed-cutoff Pareto data.

    Gaussian information includes covariance derivatives using JeffreysPrior's
    implementation, without requiring the user to remove their chosen prior.
    Bernoulli information uses differentiated logits and an analytic expectation;
    it is not the observed-data Hessian for an arbitrary nonlinear logit.
    Latent density factors and marginal hierarchical information are excluded.
    """
    refuse_ambient_float32(doing="automatic likelihood geometry")
    names = resolve_names(graph, names)
    env = evaluate(graph, values)
    if any(type(_bernoulli_base(apply_probabilistic(graph, graph.node(n), env)))
           is dist.Pareto for n in graph.observed):
        return pareto_information(graph, names, values)
    if any(
        isinstance(
            _bernoulli_base(apply_probabilistic(graph, graph.node(n), env)),
            (dist.BernoulliLogits, dist.BernoulliProbs),
        )
        for n in graph.observed
    ):
        return _bernoulli_information(graph, names, values)
    for name in graph.observed:
        if _ancestors(graph, name).intersection(graph.observed):
            raise GraphError(
                f"Gaussian expected information does not support observation-dependent "
                f"parameters on {name!r}; substituting observed parents is not the "
                "joint data expectation."
            )
    # Check the likelihood family before linearization accesses its loc.
    # Non-Gaussian observations must produce an applicability finding.
    precision = precision_at(graph, values)
    block = local_block(graph, names, values)
    design = dense_operator(block)
    refuse_single_precision(design, doing="automatic likelihood geometry")
    matrix = design.T @ _weighted_design(block, design, precision)
    spans, _ = _spans(block)
    # Always differentiate covariance here, including variance-only parameters.
    # A false depends_on_prediction annotation must not erase information.
    matrix += 2 * _log_spectrum_curvature(
        block, precision_from_graph(graph, values), {n: values[n] for n in names}, spans
    )
    return np.asarray((matrix + matrix.T) / 2)


def _geometry(matrix):
    scale = np.sqrt(np.maximum(np.diag(matrix), np.finfo(float).tiny))
    normalized = matrix / scale[:, None] / scale[None, :]
    eigenvalues = np.linalg.eigvalsh(normalized)
    rank = int(
        np.count_nonzero(
            eigenvalues > DEFAULT_RANK_RTOL * max(float(eigenvalues[-1]), 0)
        )
    )
    sign, logdet = np.linalg.slogdet(matrix)
    return rank, float(logdet) if sign > 0 and rank == matrix.shape[0] else None


def _information_scope(graph):
    """A zero direct-observation column need not be a posterior degeneracy."""
    return {
        "information_scope": "direct_observed_likelihood",
        "information_kind": "expected_fisher",
        "latent_density_factors_excluded": tuple(graph.latents),
        "hierarchical_latent_factors": tuple(
            n for n in graph.latents if _ancestors(graph, n).intersection(graph.latents)
        ),
        "evidence_terms_excluded": len(graph.evidence_terms),
        "joint_prior_excluded": graph.joint_prior is not None,
        "marginal_information": "not_assessed",
        "posterior_identifiability": "not_assessed",
    }


def analyze_preflight(runtime, task):
    """Diagnostic scheduling is once per plan, and every skipped check is filed."""
    policy, graph = task.diagnostics, runtime.graph
    if not policy.enabled:
        return (_finding("automatic_diagnostics", "disabled"),)
    findings = [
        _finding(
            "prior_policy",
            "passed",
            policy="user_specified_unchanged",
            joint_prior=type(graph.joint_prior).__name__ if graph.joint_prior else None,
        )
    ]
    information_scope = _information_scope(graph)
    if information_scope["hierarchical_latent_factors"]:
        collapse_requested = bool(dict(task.backend_options).get("collapse", False))
        collapsed = collapse_requested and runtime.sampled is not None and runtime.exact is not None
        findings.append(_finding(
            "latent_treatment", "passed",
            mode="conditional_reconstruction" if collapsed else "joint_sampling",
            sampled_parameters=tuple(graph.latents),
            conditional_density_factors=information_scope["hierarchical_latent_factors"],
            target="full_joint_likelihood_and_declared_priors",
            marginalisation_required=collapsed,
            marginal_fisher_required=False,
            marginal_jeffreys="not_assessed",
            reason="explicit_latents_do_not_require_marginalisation" if not collapsed else "requested_collapsed_route",
        ))
    try:
        points, attempts = complete_initial_values(
            graph,
            jax.random.key(193),
            task.initialization or InitializationPolicy(),
            task.budget.chains or 1,
        )
        values = points[0]
        findings.append(
            _finding(
                "initial_point",
                "passed",
                attempts=attempts,
                purpose="diagnostic_anchor_not_chain_initial_state",
                seed=193,
                anchor=tuple(
                    (
                        name,
                        tuple(np.shape(value)),
                        tuple(float(x) for x in np.real(np.asarray(value)).ravel()),
                        tuple(float(x) for x in np.imag(np.asarray(value)).ravel()),
                    )
                    for name, value in values.items()
                ),
            )
        )
    except _UNAVAILABLE as error:
        findings.append(_finding("initial_point", "unresolved", reason=str(error)))
        for code, scope in [
            ("joint_geometry", "model"),
            *[
                (f"block_{i}_jeffreys", f"block_{i}")
                for i in range(len(runtime.blocks))
            ],
        ]:
            findings.append(
                _finding(code, "unresolved", scope, reason="no_valid_diagnostic_anchor")
            )
        return tuple(findings)
    findings.append(
        _finding(
            "diagnostic_budget",
            "passed",
            max_parameters=policy.max_parameters,
            max_matrix_elements=policy.max_matrix_elements,
            max_prior_parameters=policy.max_prior_parameters,
        )
    )
    env = evaluate(graph, values)
    information_scope["observation_families"] = tuple(
        (
            n,
            type(
                _bernoulli_base(apply_probabilistic(graph, graph.node(n), env))
            ).__name__,
        )
        for n in graph.observed
    )
    data_size = 0
    for name in graph.observed:
        node = graph.node(name)
        distribution = apply_probabilistic(graph, node, env)
        data_size += int(np.prod(_observation_shape(graph, node, distribution)))
    npar = sum(np.size(v) for v in values.values())
    has_pareto = any(type(_bernoulli_base(apply_probabilistic(graph, graph.node(n), env)))
                     is dist.Pareto for n in graph.observed)
    gaussian_information = not has_pareto and not any(
        isinstance(
            _bernoulli_base(apply_probabilistic(graph, graph.node(n), env)),
            (dist.BernoulliLogits, dist.BernoulliProbs),
        )
        for n in graph.observed
    )
    certificates = {}

    def structure(names):
        names = tuple(names)
        if names not in certificates:
            certificate = gaussian_flatness_certificate(graph, names, values)
            if not gaussian_information:
                def logits(*block_values):
                    point = dict(values)
                    point.update(zip(names, block_values, strict=True))
                    environment = evaluate(graph, point)
                    return tuple(
                        (pareto_parameters(graph, n, environment) if has_pareto
                         else _bernoulli_logits(graph, n, environment)) for n in graph.observed
                    )

                derivative_evidence = derivative_semantics_evidence(
                    logits, *(jnp.asarray(values[n]) for n in names)
                )
                certificate["numerical_derivatives_trusted"] = derivative_evidence["trusted"]
                certificate["derivative_evidence"] = derivative_evidence
            certificates[names] = certificate
        return certificates[names]

    def derivatives_trusted(certificate):
        return certificate["numerical_derivatives_trusted"]

    def allowed(size):
        return (
            size <= policy.max_parameters
            and max(size * data_size, size * size) <= policy.max_matrix_elements
        )

    if not graph.latents:
        findings.append(
            _finding(
                "joint_geometry", "not_applicable", parameters=0,
                reason="no_latent_parameters", **information_scope,
            )
        )
    elif not allowed(npar):
        findings.append(
            _finding(
                "joint_geometry",
                "skipped_budget",
                parameters=npar,
                matrix_elements=max(npar * data_size, npar * npar),
            )
        )
    else:
        joint_certificate = structure(graph.latents)
        joint_derivatives_trusted = derivatives_trusted(joint_certificate)
        try:
            matrix = likelihood_information(graph, graph.latents, values)
            rank, _ = _geometry(matrix)
            # A location-only diagnostic must not prevent a supported
            # non-location family's Fisher information from being reported.
            try:
                mean_rank = identifiability(graph, at=values).rank
                mean_reason = None
            except _UNAVAILABLE as error:
                mean_rank, mean_reason = None, str(error)
            findings.append(
                _finding(
                    "joint_geometry",
                    "unresolved" if not joint_derivatives_trusted
                    else "passed" if rank == npar else "rank_deficient",
                    parameters=npar,
                    mean_rank=mean_rank,
                    mean_rank_unavailable_reason=mean_reason,
                    fisher_rank=rank,
                    scope_note="local_direct_observed_likelihood_not_global_identification",
                    rank_rtol=DEFAULT_RANK_RTOL,
                    reason=None if joint_derivatives_trusted else "unsupported_derivative_semantics",
                    structural_certificate=_certificate_value(joint_certificate),
                    **information_scope,
                )
            )
        except _UNAVAILABLE as error:
            findings.append(
                _finding(
                    "joint_geometry",
                    "unresolved",
                    reason=str(error),
                    **information_scope,
                )
            )

    for index, block in enumerate(runtime.blocks):
        scope = f"block_{index}"
        names = tuple(block.latents)
        size = sum(np.size(values[n]) for n in names)
        findings.append(
            _finding(
                f"{scope}_prior",
                "passed",
                scope,
                policy="user_specified_unchanged",
                members=names,
                distributions=tuple(
                    type(apply_probabilistic(graph, graph.node(n), env)).__name__
                    for n in names
                ),
            )
        )
        code = f"{scope}_jeffreys"
        if not allowed(size):
            findings.append(_finding(code, "skipped_budget", scope, parameters=size))
        else:
            certificate = structure(names)
            certificate_value = _certificate_value(certificate)
            block_derivatives_trusted = derivatives_trusted(certificate)
            try:
                matrix = likelihood_information(graph, names, values)
                rank, logdet = _geometry(matrix)
                if logdet is None:
                    findings.append(
                        _finding(
                            code,
                            "rank_deficient" if block_derivatives_trusted else "unresolved",
                            scope,
                            rank=rank,
                            parameters=size,
                            members=names,
                            structural_certificate=certificate_value,
                            global_flatness_proved=False,
                            reason=None if block_derivatives_trusted else "unsupported_derivative_semantics",
                            **information_scope,
                        )
                    )
                else:
                    # Compare a second support-valid block point, keeping the
                    # complement fixed. Agreement is evidence, not a proof.
                    fixed = tuple(
                        NamedArray(
                            n,
                            np.asarray(v),
                            tuple(f"axis_{i}" for i in range(np.ndim(v))),
                        )
                        for n, v in values.items()
                        if n not in names
                    )
                    probes, _ = complete_initial_values(
                        graph,
                        jax.random.key(194 + index),
                        InitializationPolicy(values=fixed),
                        1,
                    )
                    other_matrix = likelihood_information(graph, names, probes[0])
                    _, other = _geometry(other_matrix)
                    delta = None if other is None else 0.5 * (other - logdet)
                    # logdet magnifies relative matrix errors by condition.
                    # For ||A^-1 dA|| <= rho < 1, its absolute perturbation
                    # is bounded by n * -log(1-rho). Matrix assembly error
                    # below is an explicit working assumption, not a proof
                    # for arbitrary user-defined derivatives.
                    relative_error = 100 * np.finfo(float).eps * max(size, 1)
                    conditions = (
                        float(np.linalg.cond(matrix)),
                        float(np.linalg.cond(other_matrix)),
                    )
                    rhos = tuple(relative_error * c for c in conditions)
                    numerical_band = (
                        None
                        if max(rhos) >= 1
                        else float(-0.5 * size * sum(np.log1p(-rho) for rho in rhos))
                    )
                    status = (
                        "unresolved"
                        if not block_derivatives_trusted
                        else
                        "nonflat"
                        if delta is not None
                        and numerical_band is not None
                        and abs(delta) > numerical_band
                        else "unresolved"
                    )
                    # Structure is independent of sampler/prior eligibility.
                    # Numerical contradictions and rank/roundoff uncertainty
                    # remain visible even when the primal program is affine.
                    if (
                        certificate["certified"]
                        and delta is not None
                        and numerical_band is not None
                        and abs(delta) <= numerical_band
                    ):
                        status = "flat"
                    findings.append(
                        _finding(
                            code,
                            status,
                            scope,
                            members=names,
                            coordinates="model",
                            rank=rank,
                            half_logdet=0.5 * logdet,
                            half_logdet_change=delta,
                            probe_roundoff_band=numerical_band,
                            matrix_error_assumption=float(relative_error),
                            condition_numbers=conditions,
                            interpretation=(
                                "structural_affine_fixed_covariance_conditional"
                                if status == "flat"
                                else "local_observed_likelihood_probes_do_not_prove_global_flatness"
                            ),
                            flatness_evidence=(
                                "primal_jaxpr_affinity_and_covariance_independence"
                                if status == "flat"
                                else "two_local_information_probes"
                            ),
                            structural_certificate=certificate_value,
                            global_flatness_proved=status == "flat",
                            reason=(
                                "unsupported_derivative_semantics"
                                if not block_derivatives_trusted
                                else "structural_numerical_contradiction"
                                if status == "nonflat" and certificate["certified"]
                                else None
                                if status != "unresolved"
                                else "probe_information_rank_or_roundoff_unresolved"
                                if other is None or numerical_band is None
                                else "matching_local_probes_without_affine_fixed_covariance_guarantee"
                            ),
                            conditioned_on=tuple(
                                n for n in graph.latents if n not in names
                            ),
                            **information_scope,
                            prior_action="unchanged",
                        )
                    )
            except _UNAVAILABLE as error:
                findings.append(
                    _finding(
                        code,
                        "unresolved",
                        scope,
                        reason=str(error),
                        structural_certificate=certificate_value,
                        global_flatness_proved=False,
                        **information_scope,
                    )
                )
        code = f"{scope}_prior_sensitivity"
        if size > policy.max_prior_parameters or not allowed(size):
            findings.append(_finding(code, "skipped_budget", scope, parameters=size))
        else:
            try:
                from bayesmith.diagnose.sensitivity import (
                    CRITERION_SHIFT,
                    prior_sensitivity,
                )

                non_gaussian = tuple(
                    n for n in names
                    if type(_bernoulli_base(apply_probabilistic(graph, graph.node(n), env)))
                    is not dist.Normal
                )
                if non_gaussian:
                    findings.append(_finding(
                        code, "not_applicable", scope,
                        members=names, non_gaussian_priors=non_gaussian,
                        reason="gaussian_prior_perturbation_not_applicable",
                        boundary_sensitivity="not_assessed",
                        prior_action="unchanged",
                    ))
                    continue

                if (
                    graph.joint_prior is not None
                    or graph.evidence_terms
                    or any(
                        graph.node(n).observed_mask is not None for n in graph.observed
                    )
                ):
                    raise GraphError(
                        "prior sensitivity does not support joint/evidence factors or observation masks"
                    )

                report = prior_sensitivity(graph, names=names, at=values)
                verified = report.refit_converged and bool(np.all(report.verified))
                shift = float(np.max(np.abs(report.shift_sigma)))
                sensitivity_status = (
                    "unresolved"
                    if not verified
                    else "passed"
                    if shift < CRITERION_SHIFT
                    else "failed"
                )
                findings.append(
                    _finding(
                        code,
                        sensitivity_status,
                        scope,
                        max_shift_sigma=shift,
                        criterion_shift=CRITERION_SHIFT,
                        refit_converged=report.refit_converged,
                        sensitivity_verified=bool(np.all(report.verified)),
                        interpretation="measured_sensitivity_not_prior_approval",
                    )
                )
            except _UNAVAILABLE as error:
                findings.append(_finding(code, "unresolved", scope, reason=str(error)))
    collapse_runtime = getattr(runtime, "collapse_plan", runtime)
    collapsed = (
        dict(task.backend_options).get("collapse", False)
        and collapse_runtime.sampled is not None
        and collapse_runtime.exact is not None
    )
    findings.append(
        _finding(
            "sampling_order",
            "passed",
            order=(
                (tuple(collapse_runtime.sampled.latents),)
                if collapsed
                else tuple(tuple(b.latents) for b in runtime.blocks)
            ),
            rule=(
                "marginal_nuts_then_conditional_reconstruction"
                if collapsed
                else "fixed_compiler_order_latest_complement"
            ),
        )
    )
    return tuple(findings)
