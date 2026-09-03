"""Literal classified snapshot of the numerical-gate source census."""

from tests.numerical_gates.source_scan import (
    CandidateClassification,
    ManifestEntry,
)

EXPECTED_SOURCE_MANIFEST = (
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.KroneckerStructure.__init__::decision_predicate::bc9d77030723b036::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "tuple((_read_only_array(factor, ndim=2) for factor in factors))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.KroneckerStructure.__init__::decision_predicate::e91fa023925b66f9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not copied",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.KroneckerStructure.__init__::raise::26299cc85c708d19::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('KroneckerStructure needs at least one factor')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.KroneckerStructure.__init__::decision_predicate::688cce96cd9981f8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "any((factor.shape[0] == 0 or factor.shape[0] != factor.shape[1] for factor in copied))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.KroneckerStructure.__init__::compare::2d6c29fcff7eab0c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factor.shape[0] == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.KroneckerStructure.__init__::decision_predicate::2af2da82b901b77c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factor.shape[0] == 0 or factor.shape[0] != factor.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.KroneckerStructure.__init__::boolean_atom::2d6c29fcff7eab0c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factor.shape[0] == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.KroneckerStructure.__init__::compare::9c59f63722a4a579::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factor.shape[0] != factor.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.KroneckerStructure.__init__::boolean_atom::9c59f63722a4a579::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factor.shape[0] != factor.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.KroneckerStructure.__init__::raise::bb619978c1d2e6d4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Kronecker factors must be non-empty square matrices')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::compare::5f34b2b509e7693c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "array.shape[0] == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::decision_predicate::92c37852a9960996::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "array.shape[0] == 0 or array.shape[1] == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::boolean_atom::5f34b2b509e7693c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "array.shape[0] == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::compare::9316c42bfb9456cd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "array.shape[1] == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::boolean_atom::9316c42bfb9456cd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "array.shape[1] == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::raise::54dcd768eddb2754::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('FrozenProbes needs a non-empty (probes, n) array')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::decision_predicate::a5d3b66b1f22170f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.all(np.isfinite(array))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::predicate_call_atom::c06551febb069e8b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(array))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::predicate_call_atom::35b81c96644141fb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(array)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::finite_predicate::35b81c96644141fb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(array)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.FrozenProbes.__init__::raise::4c36982aefd3caab::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('FrozenProbes values must all be finite')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::decision_predicate::74183c3f059a2e1f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_read_only_array(left, ndim=2)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::compare::26af4950c7d14dea::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::decision_predicate::26af4950c7d14dea::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::decision_predicate::8d89973d6192af5a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_read_only_array(right, ndim=2)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::compare::b3b1da488848a9de::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "candidate.dtype == left_array.dtype",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::decision_predicate::1b3da62f7b90cc45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "candidate.dtype == left_array.dtype and candidate.shape == left_array.shape and (candidate.tobytes() == left_array.tobytes())",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::boolean_atom::b3b1da488848a9de::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "candidate.dtype == left_array.dtype",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::compare::50d9cfb1eeeaf6db::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "candidate.shape == left_array.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::boolean_atom::50d9cfb1eeeaf6db::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "candidate.shape == left_array.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::compare::a53a9232b15f461b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "candidate.tobytes() == left_array.tobytes()",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::boolean_atom::a53a9232b15f461b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "candidate.tobytes() == left_array.tobytes()",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::decision_predicate::d1ab9bd3304acfc5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "same_representation",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::compare::511371062b04a65e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left_array.shape[1] != right_array.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::decision_predicate::511371062b04a65e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left_array.shape[1] != right_array.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LowRankFactors.__init__::raise::63848ae09485fbf9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('low-rank factors must have the same column count')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::decision_predicate::08e52b1635749332::0",
        CandidateClassification.NUMERICAL_GATE,
        "any((value < 0 for value in integer_fields))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::compare::d56ec5b9fc884113::0",
        CandidateClassification.NUMERICAL_GATE,
        "value < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::raise::e3e6012da7fc0166::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('ladder size and rank thresholds must be non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::decision_predicate::f29cb1f137f9131a::0",
        CandidateClassification.NUMERICAL_GATE,
        "not 0.0 <= self.low_rank_fraction <= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::compare::fa2fc329c066f2fc::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 <= self.low_rank_fraction <= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::raise::0140eccb5a83dac9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('low_rank_fraction must lie in [0, 1]')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::decision_predicate::50468fbb991e3009::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(self.structure_rtol) or not np.isfinite(self.structure_atol) or self.structure_rtol < 0.0 or (self.structure_atol < 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::boolean_atom::b56b690ffc057990::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(self.structure_rtol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::predicate_call_atom::433cf901a0d411f0::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(self.structure_rtol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::finite_predicate::433cf901a0d411f0::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(self.structure_rtol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::boolean_atom::008a5c33809fc816::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(self.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::predicate_call_atom::b0aea59f157c766f::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(self.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::finite_predicate::b0aea59f157c766f::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(self.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::compare::38550887364ae38c::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.structure_rtol < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::boolean_atom::38550887364ae38c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.structure_rtol < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::compare::baa72a1b6357ac36::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.structure_atol < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::boolean_atom::baa72a1b6357ac36::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.structure_atol < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::raise::a8cd29a1adbd03ba::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('structure_rtol and structure_atol must be finite and non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::decision_predicate::2b61f5f86861b6c3::0",
        CandidateClassification.NUMERICAL_GATE,
        "not _is_positive_definite(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::predicate_call_atom::2f5d27d8275ca569::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_positive_definite(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::raise::3f96309c5446e1c0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Lambda must be symmetric positive definite')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::compare::0182b6609a56afdd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen_probes is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::decision_predicate::04a99800aa615570::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen_probes is not None and type(frozen_probes) is not FrozenProbes",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::boolean_atom::0182b6609a56afdd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen_probes is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::compare::17c875160753b86c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "type(frozen_probes) is not FrozenProbes",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::boolean_atom::17c875160753b86c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "type(frozen_probes) is not FrozenProbes",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::raise::abf5aa302d250606::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise TypeError('frozen_probes must be an exact FrozenProbes bytes-backed instance')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::compare::fd39062c1e2458c5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "exact_power_traces is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::decision_predicate::fd39062c1e2458c5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "exact_power_traces is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::compare::e6cd3e456539473f::0",
        CandidateClassification.NUMERICAL_GATE,
        "ndim is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::decision_predicate::0020fd8302214f01::0",
        CandidateClassification.NUMERICAL_GATE,
        "ndim is not None and array.ndim != ndim",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::boolean_atom::e6cd3e456539473f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "ndim is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::compare::0023146ac52e33f1::0",
        CandidateClassification.NUMERICAL_GATE,
        "array.ndim != ndim",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::boolean_atom::0023146ac52e33f1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "array.ndim != ndim",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::raise::6768e0d765137dc1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'expected a {ndim}-dimensional array, got shape {array.shape}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::compare::4b9b4c4f077e83f4::0",
        CandidateClassification.NUMERICAL_GATE,
        "array.ndim not in (1, 2)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::decision_predicate::4b9b4c4f077e83f4::0",
        CandidateClassification.NUMERICAL_GATE,
        "array.ndim not in (1, 2)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::raise::3213838abd16f5b1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'matrix inputs must be one- or two-dimensional, got {array.shape}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::decision_predicate::b86030f69c6dc661::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.issubdtype(array.dtype, np.floating)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::compare::e3cdf1c865f8ddca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "array.dtype.type is np.float16",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::decision_predicate::e3cdf1c865f8ddca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "array.dtype.type is np.float16",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::compare::b2b15f1786d32931::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "array.dtype.type not in (np.float32, np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::decision_predicate::b2b15f1786d32931::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "array.dtype.type not in (np.float32, np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::decision_predicate::a5d3b66b1f22170f::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.all(np.isfinite(array))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::predicate_call_atom::c06551febb069e8b::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(array))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::predicate_call_atom::35b81c96644141fb::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(array)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::finite_predicate::35b81c96644141fb::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(array)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::raise::52e0550c3c40ca4b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('matrix inputs must be finite')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::compare::e14a9566bba1b582::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.shape != perturb.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::decision_predicate::293b4d9f83cc6444::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.shape != perturb.shape or lam.ndim not in (1, 2)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::boolean_atom::e14a9566bba1b582::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.shape != perturb.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::compare::67694e37ebc9c438::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.ndim not in (1, 2)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::boolean_atom::67694e37ebc9c438::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.ndim not in (1, 2)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::raise::014da332549f92f1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Lambda and perturbation must have the same diagonal-vector or square-matrix shape')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::compare::b05c1f89c41af0d1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.ndim == 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::decision_predicate::1e7216663c503612::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.ndim == 2 and lam.shape[0] != lam.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::boolean_atom::b05c1f89c41af0d1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.ndim == 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::compare::bfbadb4a75f76b2b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.shape[0] != lam.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::boolean_atom::bfbadb4a75f76b2b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.shape[0] != lam.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::raise::c63616a495dc5e72::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Lambda and perturbation matrices must be square')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::decision_predicate::a14e7fce0e95f8d9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.issubdtype(computation_dtype, np.floating)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::compare::bc940b902439ac18::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "computation_dtype.type is np.float16",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::decision_predicate::bc940b902439ac18::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "computation_dtype.type is np.float16",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::compare::6fe97bc98ed163f3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "computation_dtype.type not in (np.float32, np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matrix_pair::decision_predicate::6fe97bc98ed163f3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "computation_dtype.type not in (np.float32, np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._dense::compare::de7dc758f1e0c805::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._dense::decision_predicate::de7dc758f1e0c805::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matching_factor_reconstruction::decision_predicate::b678026387cb59b1::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(canonical, value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matching_factor_reconstruction::predicate_call_atom::b678026387cb59b1::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(canonical, value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matching_factor_reconstruction::decision_predicate::cba17196a4c6f789::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(reconstructed, value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matching_factor_reconstruction::predicate_call_atom::cba17196a4c6f789::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(reconstructed, value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::f1bb43905c768163::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left[:, column]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::6975fb6a2d44a87b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right[:, column]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::3745d19715f85de9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.max(np.abs(left_column), initial=0.0))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::2020d78d8340fd64::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.max(np.abs(right_column), initial=0.0))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::compare::b482308942d663c1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left_maximum == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::35a807d2a59d834e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left_maximum == 0.0 or right_maximum == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::boolean_atom::b482308942d663c1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left_maximum == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::compare::1d04a6b85048b023::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right_maximum == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::boolean_atom::1d04a6b85048b023::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right_maximum == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::e38f9acd65ee490a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.frexp(left_maximum)[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::e44455ec438d6f37::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.frexp(right_maximum)[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::808d46cb6a01f633::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "(right_exponent - left_exponent) // 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::f43d8a2409d16473::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.ldexp(left_column, shift)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::70ffe0129fd38e18::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.ldexp(right_column, -shift)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::9746a632caec8b8a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.ldexp(scaled_left, -shift)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::e6e5d1a290a8e26c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.ldexp(scaled_right, shift)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::0a4170fb4dffee95::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(scaled_left)) and np.all(np.isfinite(scaled_right)) and np.array_equal(restored_left, left_column) and np.array_equal(restored_right, right_column)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::boolean_atom::508f419c8ad7bceb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(scaled_left))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::508f419c8ad7bceb::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(scaled_left))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::633c4ba7bc098a19::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(scaled_left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::finite_predicate::633c4ba7bc098a19::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(scaled_left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::boolean_atom::ff9f64daa7a7b618::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(scaled_right))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::ff9f64daa7a7b618::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(scaled_right))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::09a90b48970dbb22::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(scaled_right)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::finite_predicate::09a90b48970dbb22::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(scaled_right)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::boolean_atom::c62727bc99ba15f1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(restored_left, left_column)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::c62727bc99ba15f1::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(restored_left, left_column)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::boolean_atom::78d67bade0ec5240::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(restored_right, right_column)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::78d67bade0ec5240::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(restored_right, right_column)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::d03714ef1cdf7dcb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not exactly_reversible",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::raise::60b2c8a500fe9614::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('power-of-two factor balancing would underflow a nonzero entry or produce a non-finite value')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::raise::9c02c094bcd1faf2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(failure) from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::decision_predicate::3173248810b47e31::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.all(np.isfinite(summed)) or not np.all(np.isfinite(error))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::boolean_atom::99eaf5b9342bcd6b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.all(np.isfinite(summed))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::predicate_call_atom::8759e32180530bd3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(summed))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::predicate_call_atom::f1116af0f29a0fad::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(summed)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::finite_predicate::f1116af0f29a0fad::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(summed)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::boolean_atom::38c55da356f3a1d8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.all(np.isfinite(error))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::predicate_call_atom::0e428760cd76c29c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(error))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::predicate_call_atom::96c05eedbef96c67::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(error)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::finite_predicate::96c05eedbef96c67::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(error)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._two_sum_error::raise::23f09c8490aad598::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(failure)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._roundoff_gamma::compare::8da450fabbb947ef::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "product >= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._roundoff_gamma::decision_predicate::8da450fabbb947ef::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "product >= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::6873ebff50e05167::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors.left.shape[0] != value.shape[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::a47e36633e9889b3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors.left.shape[0] != value.shape[0] or factors.right.shape[0] != value.shape[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::6873ebff50e05167::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors.left.shape[0] != value.shape[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::cb47f1fb0dc2edb2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors.right.shape[0] != value.shape[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::cb47f1fb0dc2edb2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors.right.shape[0] != value.shape[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::a98f1b63723acc22::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('low-rank factor row counts must equal perturbation size')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::aeaea49a92e5294d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::aeaea49a92e5294d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::62f195ddbf944dde::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('dense low-rank factors require a dense perturbation')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::23b6ce39bd710a70::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.result_type(value.dtype, lambda_matrix.dtype)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::bea3de6981581b3b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.issubdtype(target_dtype, np.floating)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::aae44f516d01909b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.dtype(float)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::25f011811136b16e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "target_dtype.type is np.float16",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::25f011811136b16e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "target_dtype.type is np.float16",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::1bcae7dd1971faa3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.dtype(np.float32)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::6718c6bcaedacb46::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "target_dtype.type not in (np.float32, np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::6718c6bcaedacb46::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "target_dtype.type not in (np.float32, np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::c46b9b6f06697401::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.dtype(np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::79312eb66c1eff1a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.asarray(value, dtype=target_dtype)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::bb3549dbc4e647a7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.asarray(lambda_matrix, dtype=target_dtype)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::09ea5a346dd111bb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "target_lam.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::e0a2540bdfde51d2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "target_lam.ndim != 2 or target_lam.shape != target_perturbation.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::09ea5a346dd111bb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "target_lam.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::3f422d996f660d48::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "target_lam.shape != target_perturbation.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::3f422d996f660d48::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "target_lam.shape != target_perturbation.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::87c1898bddaa4890::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('dense low-rank factors require a matching dense Lambda matrix')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::3232e2c214780c58::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.result_type(target_dtype, factors.left.dtype, factors.right.dtype)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::67ad7eeeeef81d69::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "work_dtype.type not in (np.float32, np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::67ad7eeeeef81d69::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "work_dtype.type not in (np.float32, np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::c46b9b6f06697401::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.dtype(np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::93151a19bdc20e98::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.asarray(target_perturbation, dtype=work_dtype)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::221b79c4c5af5879::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.asarray(target_lam, dtype=work_dtype)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::numerical_premise_call::f74f734315c04d72::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_two_sum_error(target_lam, target_perturbation)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::c649a40eefd965d3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.asarray(target_sigma, dtype=work_dtype)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::350166ec2111dbc9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.asarray(target_addition_error, dtype=work_dtype)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::e4ca448632bc2b06::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.qr(balanced_left, mode='reduced')[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::cb70e910e0bd0ddb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.qr(balanced_right, mode='reduced')[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::709e6ee12e3e96e3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left_basis.T @ perturbation @ right_basis",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::07d5b6fc5f874369::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left_basis @ core @ right_basis.T",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::982a80f5648ced01::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('factor projection could not be evaluated with finite QR arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::ec16f32088da528c::0",
        CandidateClassification.NUMERICAL_GATE,
        "not (np.all(np.isfinite(left_basis)) and np.all(np.isfinite(right_basis)) and np.all(np.isfinite(core)) and np.all(np.isfinite(projected)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::faa44f657bf8cd00::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(left_basis)) and np.all(np.isfinite(right_basis)) and np.all(np.isfinite(core)) and np.all(np.isfinite(projected))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::bdef49efc73fa379::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(left_basis))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::bdef49efc73fa379::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(left_basis))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::46056fb4c1718479::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(left_basis)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::46056fb4c1718479::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(left_basis)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::db5916e25d36ef64::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(right_basis))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::db5916e25d36ef64::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(right_basis))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::854832823b821fe4::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(right_basis)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::854832823b821fe4::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(right_basis)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::c2294f94e70821ca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(core))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::c2294f94e70821ca::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(core))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::f26acd7566369a93::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(core)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::f26acd7566369a93::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(core)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::039c47cc071f1e8e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(projected))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::039c47cc071f1e8e::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(projected))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::6e2471591892211c::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(projected)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::6e2471591892211c::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(projected)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::29b947cc4bfa8fbd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('factor projection could not be evaluated with finite QR arithmetic')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::0d4f7018485ffd66::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "perturbation - projected - addition_error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::3a5cb9b42b0eba11::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.diag(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::28bf0bba9f19530e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(lam, np.diag(diagonal))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::28bf0bba9f19530e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(lam, np.diag(diagonal))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::linalg_exception_premise::ef2bde635a089a4b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "try:\n    with np.errstate(over='raise', divide='raise', invalid='raise'):\n        if lambda_is_diagonal:\n            square_root = np.sqrt(diagonal)\n            whitened_residual = residual / square_root[:, None] / square_root[None, :]\n            whitened_sigma = sigma / square_root[:, None] / square_root[None, :]\n        else:\n            lambda_factor = np.linalg.cholesky(lam)\n            left_whitened_residual = np.linalg.solve(lambda_factor, residual)\n            whitened_residual = np.linalg.solve(lambda_factor, left_whitened_residual.T).T\n            left_whitened_sigma = np.linalg.solve(lambda_factor, sigma)\n            whitened_sigma = np.linalg.solve(lambda_factor, left_whitened_sigma.T).T\n    symmetric_whitened_sigma = _symmetric_roundoff_representative(whitened_sigma, operation='factor projection whitened-Sigma symmetrization')\n    smallest_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric_whitened_sigma)))\nexcept (FloatingPointError, np.linalg.LinAlgError, TypeError, ValueError) as error:\n    raise ValueError('factor projection could not whiten Lambda and Sigma with finite supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::b56e5e39c83bbabb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_is_diagonal",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::0a7b1eb6fc66f770::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "residual / square_root[:, None] / square_root[None, :]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::45505338170e1bbc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma / square_root[:, None] / square_root[None, :]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::2fec347a8d7a5962::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.cholesky(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::linalg_call_atom::2fec347a8d7a5962::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.cholesky(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::b3559858413770d6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.solve(lambda_factor, residual)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::860d0a751443327b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.solve(lambda_factor, left_whitened_residual.T).T",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::5b075594afb752f3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.solve(lambda_factor, sigma)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::434afb49dc82ddc1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.solve(lambda_factor, left_whitened_sigma.T).T",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::9b45bfbc80f88f00::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_symmetric_roundoff_representative(whitened_sigma, operation='factor projection whitened-Sigma symmetrization')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::2d3f4669ed1752b6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.min(np.linalg.eigvalsh(symmetric_whitened_sigma)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::0c54b2a98f87b659::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('factor projection could not whiten Lambda and Sigma with finite supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::baf0df33f0add114::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.all(np.isfinite(whitened_residual))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::218cd3ac905664f3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(whitened_residual))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::1a18d3794e73422e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(whitened_residual)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::1a18d3794e73422e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(whitened_residual)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::4e6766178314d088::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.inf",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::01f755473a4e5ed9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.linalg.norm(whitened_residual, ord=2))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::260516798af6c318::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('factor projection spectral norm could not be evaluated with supported finite arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::547b40df01c853aa::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.inf if smallest_eigenvalue <= 0.0 or not np.isfinite(smallest_eigenvalue) else residual_norm / smallest_eigenvalue",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::386f873e56933cc1::0",
        CandidateClassification.NUMERICAL_GATE,
        "smallest_eigenvalue <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::3f267babe0743274::0",
        CandidateClassification.NUMERICAL_GATE,
        "smallest_eigenvalue <= 0.0 or not np.isfinite(smallest_eigenvalue)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::386f873e56933cc1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "smallest_eigenvalue <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::0ee73a7da3fdfeec::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(smallest_eigenvalue)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::92740bdbc4d0e19b::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(smallest_eigenvalue)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::92740bdbc4d0e19b::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(smallest_eigenvalue)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::f95c69c3ae01c17a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.sqrt(float(np.finfo(target_dtype).eps))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::c5bdac7c4db9ced8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "-float(value.shape[0]) * math.log1p(-eta) if np.isfinite(eta) and 0.0 <= eta < 1.0 else math.inf",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::bb4398f0aaf882ae::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(eta) and 0.0 <= eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::2223debe39afb9d2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::2223debe39afb9d2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::2223debe39afb9d2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::e12255894c7c6961::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "0.0 <= eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::e12255894c7c6961::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "0.0 <= eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::58441b26c20ee398::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.isfinite(eta) and eta < 1.0 and np.isfinite(log_error_bound) and (log_error_bound <= ceiling))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::58441b26c20ee398::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.isfinite(eta) and eta < 1.0 and np.isfinite(log_error_bound) and (log_error_bound <= ceiling))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::22296f4e1c2def63::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(eta) and eta < 1.0 and np.isfinite(log_error_bound) and (log_error_bound <= ceiling)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::2223debe39afb9d2::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::2223debe39afb9d2::1",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::2223debe39afb9d2::1",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::5dfa833658bbcfa3::0",
        CandidateClassification.NUMERICAL_GATE,
        "eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::5dfa833658bbcfa3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::8bfb56ffb90db4b0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::8bfb56ffb90db4b0::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::8bfb56ffb90db4b0::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::c30cd037d5180310::0",
        CandidateClassification.NUMERICAL_GATE,
        "log_error_bound <= ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::c30cd037d5180310::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "log_error_bound <= ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::7e932b6c58cf3309::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "int(value.shape[0])",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::ed47b4cee9c4c650::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "int(core.shape[0])",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::48a8df4f0e43603f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.finfo(work_dtype).eps)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::b56e5e39c83bbabb::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_is_diagonal",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::0deac7d9ac12ff9a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.inf if smallest_diagonal == 0.0 else float(np.max(diagonal_magnitudes, initial=0.0)) / smallest_diagonal",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::74bc8a09ce0da9c7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "smallest_diagonal == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::74bc8a09ce0da9c7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "smallest_diagonal == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::4e6766178314d088::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.inf",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::5f628b81ef6406fb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('determinant-lemma base logarithms require finite supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::89a344596de6b123::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "2.0 * work_eps * math.fsum((max(1.0, abs(float(item))) for item in diagonal_logs))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::clamp_selector::ec91cf0157ae95d2::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(1.0, abs(float(item)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::ab59ebbb79f78f47::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "work_eps / (1.0 - work_eps)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::fc419b00dba1dc91::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.linalg.cond(lam))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::a4bd5d7358d3653a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('determinant-lemma base condition could not be measured with supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::628c8c528b372e8a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "1.0 / math.sqrt(work_eps)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::36360dfcbffdb3e0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_roundoff_gamma(3 * n, work_dtype) * base_condition",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::af719d0e96a7a621::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "-float(n) * math.log1p(-base_solve_eta) if np.isfinite(base_solve_eta) and 0.0 <= base_solve_eta < 1.0 else math.inf",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::f5cf9fa84c4b9ba2::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(base_solve_eta) and 0.0 <= base_solve_eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::f077a4f9072830b5::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(base_solve_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::f077a4f9072830b5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(base_solve_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::f077a4f9072830b5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(base_solve_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::607d8e28d211c1e8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "0.0 <= base_solve_eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::607d8e28d211c1e8::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 <= base_solve_eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::54522a7de58a3d64::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.log(np.diag(lambda_factor))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::19200189337a3bd2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "4.0 * work_eps * math.fsum((max(1.0, abs(2.0 * float(item))) for item in factor_diagonal_logs))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::clamp_selector::7cb0b7a37d7c9776::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(1.0, abs(2.0 * float(item)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::f8d19bcd9802406a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.fsum((base_factorization_log_bound, base_log_roundoff_bound))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::40e27fe2adafafbf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.isfinite(base_condition) and base_condition < base_condition_ceiling and np.isfinite(base_log_error_bound) and (base_log_error_bound <= ceiling))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::40e27fe2adafafbf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.isfinite(base_condition) and base_condition < base_condition_ceiling and np.isfinite(base_log_error_bound) and (base_log_error_bound <= ceiling))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::9bdf5a54d04a04ad::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(base_condition) and base_condition < base_condition_ceiling and np.isfinite(base_log_error_bound) and (base_log_error_bound <= ceiling)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::3e2925807fea5201::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(base_condition)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::3e2925807fea5201::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(base_condition)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::3e2925807fea5201::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(base_condition)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::8979ff3006924922::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "base_condition < base_condition_ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::8979ff3006924922::0",
        CandidateClassification.NUMERICAL_GATE,
        "base_condition < base_condition_ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::0520942938172abf::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(base_log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::0520942938172abf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(base_log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::0520942938172abf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(base_log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::7eb5677a4b7f505c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "base_log_error_bound <= ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::7eb5677a4b7f505c::0",
        CandidateClassification.NUMERICAL_GATE,
        "base_log_error_bound <= ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::b56e5e39c83bbabb::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_is_diagonal",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::4c03e9330f087926::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left_basis / diagonal[:, None]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::92067ffd15d3be1e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "work_eps / (1.0 - work_eps) * np.abs(preconditioned_left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::b8ed6a1d906c4fa3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.solve(lam, left_basis)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::8f2cda988d9ca730::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::13537f024152319b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(base_solve_eta) or base_solve_eta >= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::61a9a92fb3e0a23d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(base_solve_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::f077a4f9072830b5::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(base_solve_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::f077a4f9072830b5::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(base_solve_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::f6ad2f5bcef4debb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "base_solve_eta >= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::f6ad2f5bcef4debb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "base_solve_eta >= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::f63f51221cacb3c0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right_basis.T @ preconditioned_left",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::numerical_premise_call::21af423bde83efa1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_two_sum_error(np.eye(rank, dtype=work_dtype), correction, operation='reduced determinant-lemma addition')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::17a185b14abfb088::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('reduced determinant-lemma arithmetic could not be evaluated with finite supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::42c98056e125795c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not (np.all(np.isfinite(preconditioned_left)) and np.all(np.isfinite(preconditioned_bases)) and np.all(np.isfinite(correction)) and np.all(np.isfinite(reduced)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::cf19c18b2c6a1822::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(preconditioned_left)) and np.all(np.isfinite(preconditioned_bases)) and np.all(np.isfinite(correction)) and np.all(np.isfinite(reduced))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::f9285b455e953cc1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(preconditioned_left))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::f9285b455e953cc1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(preconditioned_left))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::91f9e607b7117b96::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(preconditioned_left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::91f9e607b7117b96::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(preconditioned_left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::225ac5bf2a7685bf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(preconditioned_bases))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::225ac5bf2a7685bf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(preconditioned_bases))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::6eb01b530d5e97d7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(preconditioned_bases)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::6eb01b530d5e97d7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(preconditioned_bases)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::358900bcb96c21c4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(correction))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::358900bcb96c21c4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(correction))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::9fceeafc586a70d0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(correction)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::9fceeafc586a70d0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(correction)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::1d2bea20abc7db0b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(reduced))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::1d2bea20abc7db0b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(reduced))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::bf23d7513d3e3333::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::bf23d7513d3e3333::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::caef32d80f33dce9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('reduced determinant-lemma arithmetic requires finite matrices')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::cb7217eef1037967::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rank == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::cb7217eef1037967::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rank == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::8f2cda988d9ca730::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::8f2cda988d9ca730::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::624c0d6fb9bc114c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::a77ce679a39d5861::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::624c0d6fb9bc114c::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::a77ce679a39d5861::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::624c0d6fb9bc114c::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::a77ce679a39d5861::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::42e29a84dea5b957::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_roundoff_gamma(n, work_dtype) * (np.abs(right_basis.T) @ np.abs(preconditioned_left))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::1a64262249a3e534::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "preconditioned_left_error is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::1a64262249a3e534::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "preconditioned_left_error is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::8f2cda988d9ca730::3",
        CandidateClassification.ORDINARY_VALIDATION,
        "None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::9dbd3462ad56b8bf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.abs(right_basis.T) @ preconditioned_left_error + first_roundoff_envelope",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::9f2d432a2a8dc248::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_roundoff_gamma(rank, work_dtype) * (np.abs(preconditioned_bases) @ np.abs(core))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::6a6527d71c562d8c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "first_error_envelope is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::6a6527d71c562d8c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "first_error_envelope is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::8f2cda988d9ca730::4",
        CandidateClassification.ORDINARY_VALIDATION,
        "None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::5d37d206ce018491::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "correction_error_norm + float(np.linalg.norm(reduced_addition_error, ord=2))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::297cc8976d020523::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "first_error_envelope @ np.abs(core) + correction_roundoff_envelope",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::12b1e2616b31ab49::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "correction_error_envelope + np.abs(reduced_addition_error)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::d041f5999c844735::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.linalg.norm(reduced_formation_envelope, ord=2))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::7c0ddd9c74ceb8d0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.svd(reduced, compute_uv=False)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::ac19f3728a3dcd84::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.linalg.cond(reduced))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::9f97220e9659347b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('reduced determinant-lemma conditioning could not be measured with supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::951bb6041c4c881b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_roundoff_gamma(3 * rank, work_dtype) * float(np.linalg.norm(np.abs(reduced), ord=2))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::64e6c47c58696496::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "max(0.0, float(singular_values[-1]) - singular_value_roundoff_margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::clamp_selector::64e6c47c58696496::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(0.0, float(singular_values[-1]) - singular_value_roundoff_margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::fd0eeee696f63509::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(reduced, np.diag(np.diag(reduced)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::fd0eeee696f63509::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(reduced, np.diag(np.diag(reduced)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::b2d59cfcd5c191e2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_formation_envelope is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::7bed19ad22f554fb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_formation_envelope is not None and np.array_equal(reduced_formation_envelope, np.diag(np.diag(reduced_formation_envelope)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::b2d59cfcd5c191e2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_formation_envelope is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::3b886581eed8760f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(reduced_formation_envelope, np.diag(np.diag(reduced_formation_envelope)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::3b886581eed8760f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(reduced_formation_envelope, np.diag(np.diag(reduced_formation_envelope)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::d1d1c705b6e84dc9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_is_diagonal and envelope_is_diagonal",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::246fd12c2d420b21::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_is_diagonal",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::3064e140734a1458::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "envelope_is_diagonal",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::8f2cda988d9ca730::5",
        CandidateClassification.ORDINARY_VALIDATION,
        "None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::8f2cda988d9ca730::6",
        CandidateClassification.ORDINARY_VALIDATION,
        "None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::914166ea4bf3c547::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.any(reduced_diagonal == 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::914166ea4bf3c547::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.any(reduced_diagonal == 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::3428ad26fcf63f27::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_diagonal == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::5a80459b02463f53::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.max(relative_diagonal_error, initial=0.0))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::a023074a3eba1093::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.prod(np.sign(reduced_diagonal)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::0f7dbea9e949f40c::0",
        CandidateClassification.NUMERICAL_GATE,
        "reduced_sign > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::70ad0922d1407890::0",
        CandidateClassification.NUMERICAL_GATE,
        "reduced_sign > 0.0 and np.all(np.isfinite(relative_diagonal_error)) and np.all(relative_diagonal_error < 1.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::0f7dbea9e949f40c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_sign > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::27dc5722290ed8d0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(relative_diagonal_error))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::27dc5722290ed8d0::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(relative_diagonal_error))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::5312fdea367be588::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(relative_diagonal_error)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::5312fdea367be588::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(relative_diagonal_error)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::1b033f6c18a10c46::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(relative_diagonal_error < 1.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::1b033f6c18a10c46::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(relative_diagonal_error < 1.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::4fa526a712819567::0",
        CandidateClassification.NUMERICAL_GATE,
        "relative_diagonal_error < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::clamp_selector::8de831a17eda072f::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(1.0, abs(math.log(float(item))))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::1b951612d196bf44::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.fsum((formation_log_bound, diagonal_log_roundoff))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::4e6766178314d088::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.inf",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::numerical_premise_call::47280089f3523be7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_two_sum_error(reduced, -reconstructed_reduced, operation='reduced QR reconstruction subtraction')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::numerical_premise_call::2168ba23e23146f7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_two_sum_error(orthogonal_product, -np.eye(rank, dtype=work_dtype), operation='reduced QR orthogonality subtraction')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::74bb924776593740::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.linalg.slogdet(reduced_q)[0])",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::raise::bcc30ea8c2f723e7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('reduced determinant-lemma QR certificate could not be evaluated with finite supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::a5a9bd05c56725ef::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.abs(qr_residual) + np.abs(qr_addition_error) + _roundoff_gamma(rank, work_dtype) * (np.abs(reduced_q) @ np.abs(reduced_r))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::513f372ef10e6bc8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "float(np.linalg.norm(qr_reconstruction_envelope, ord=2))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::8d1ff92032c6ac80::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.inf if reduced_smallest_singular <= 0.0 else reduced_formation_error_norm / reduced_smallest_singular",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::981773214424b929::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_smallest_singular <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::981773214424b929::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_smallest_singular <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::67a08e45e21a6bfc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.inf if reduced_smallest_singular <= 0.0 else qr_reconstruction_error_norm / reduced_smallest_singular",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::981773214424b929::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_smallest_singular <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::981773214424b929::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_smallest_singular <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::0e6d5b1bafccee4b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_formation_eta + qr_reconstruction_eta",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::17ad8b98402612fb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.diag(reduced_r)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::7fb8cb1cb472e1e7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "q_sign * float(np.prod(np.sign(reduced_r_diagonal)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::0f7dbea9e949f40c::1",
        CandidateClassification.NUMERICAL_GATE,
        "reduced_sign > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::a1dcce8aab2a2ac4::0",
        CandidateClassification.NUMERICAL_GATE,
        "reduced_sign > 0.0 and np.isfinite(reduced_eta) and (0.0 <= reduced_eta < 1.0) and np.isfinite(orthogonality_eta) and (0.0 <= orthogonality_eta < 1.0) and np.all(reduced_r_diagonal != 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::0f7dbea9e949f40c::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_sign > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::10bd024c3347566e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::10bd024c3347566e::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(reduced_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::10bd024c3347566e::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(reduced_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::5df229387047e996::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 <= reduced_eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::5df229387047e996::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "0.0 <= reduced_eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::88a2e3998c62924a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(orthogonality_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::88a2e3998c62924a::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(orthogonality_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::88a2e3998c62924a::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(orthogonality_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::c85350bec59e195e::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 <= orthogonality_eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::c85350bec59e195e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "0.0 <= orthogonality_eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::663cd7e7e74271e5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(reduced_r_diagonal != 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::663cd7e7e74271e5::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(reduced_r_diagonal != 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::b5bc8c97b914c9dd::0",
        CandidateClassification.NUMERICAL_GATE,
        "reduced_r_diagonal != 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::clamp_selector::7b75eda18a3141a7::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(1.0, abs(math.log(abs(float(item)))))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::648d349a2f663aa5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.fsum((matrix_log_bound, orthogonality_log_bound, triangular_log_roundoff))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::4e6766178314d088::3",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.inf",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::05ab4609c3af460e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.isfinite(reduced_eta) and reduced_eta < 1.0 and np.isfinite(reduced_condition) and (reduced_sign > 0.0) and np.isfinite(reduced_log_error_bound) and (reduced_log_error_bound <= ceiling))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::05ab4609c3af460e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.isfinite(reduced_eta) and reduced_eta < 1.0 and np.isfinite(reduced_condition) and (reduced_sign > 0.0) and np.isfinite(reduced_log_error_bound) and (reduced_log_error_bound <= ceiling))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::ac3abec363b6a864::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_eta) and reduced_eta < 1.0 and np.isfinite(reduced_condition) and (reduced_sign > 0.0) and np.isfinite(reduced_log_error_bound) and (reduced_log_error_bound <= ceiling)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::10bd024c3347566e::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::10bd024c3347566e::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::10bd024c3347566e::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_eta)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::6be5f5966e634285::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::6be5f5966e634285::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_eta < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::cd86bcd1e6590311::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_condition)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::cd86bcd1e6590311::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_condition)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::cd86bcd1e6590311::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_condition)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::0f7dbea9e949f40c::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_sign > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::0f7dbea9e949f40c::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_sign > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::3caaa70c6f98af42::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::3caaa70c6f98af42::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::3caaa70c6f98af42::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(reduced_log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::e301d0466e22e0bf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_log_error_bound <= ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::e301d0466e22e0bf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_log_error_bound <= ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::028e8f3d00d35aa8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.fsum((log_error_bound, base_log_error_bound, reduced_log_error_bound))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::4c4a9952327f74fc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.isfinite(total_log_error_bound) and total_log_error_bound <= ceiling)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::4c4a9952327f74fc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.isfinite(total_log_error_bound) and total_log_error_bound <= ceiling)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::14853fc7855c8120::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(total_log_error_bound) and total_log_error_bound <= ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::32ca60671f93bf91::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(total_log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::32ca60671f93bf91::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(total_log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::32ca60671f93bf91::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(total_log_error_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::db227b7e3fb87859::0",
        CandidateClassification.NUMERICAL_GATE,
        "total_log_error_bound <= ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::db227b7e3fb87859::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "total_log_error_bound <= ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::45a5324bb5d23629::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not reconstruction_matches",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::c1af0cfb95f74aa0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not projection_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::80e092eff388d601::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not base_arithmetic_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::c25c491f19ca7c08::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not reduced_arithmetic_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::3d65fdcaf1b89afa::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not total_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::b7ac1b6bf18ed04d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(reconstruction_matches and projection_valid and base_arithmetic_valid and reduced_arithmetic_valid and total_valid)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::4e98f5a96307835b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reconstruction_matches and projection_valid and base_arithmetic_valid and reduced_arithmetic_valid and total_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::45fd505b81019a42::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reconstruction_matches",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::fb7feb3036e3770a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "projection_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::d76bb5655e2de47a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "base_arithmetic_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::2344a3f652bb85ad::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_arithmetic_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::6e1e1de2e3623f34::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "total_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::compare::469cf6ad6cae420a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::decision_predicate::469cf6ad6cae420a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::compare::6873ebff50e05167::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors.left.shape[0] != value.shape[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::decision_predicate::a47e36633e9889b3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors.left.shape[0] != value.shape[0] or factors.right.shape[0] != value.shape[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::boolean_atom::6873ebff50e05167::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors.left.shape[0] != value.shape[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::compare::cb47f1fb0dc2edb2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors.right.shape[0] != value.shape[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::boolean_atom::cb47f1fb0dc2edb2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors.right.shape[0] != value.shape[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::raise::a98f1b63723acc22::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('low-rank factor row counts must equal perturbation size')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::compare::aeaea49a92e5294d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::decision_predicate::aeaea49a92e5294d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::raise::62f195ddbf944dde::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('dense low-rank factors require a dense perturbation')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::compare::af3fbecb038661ad::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_matrix is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::decision_predicate::af3fbecb038661ad::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_matrix is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::decision_predicate::45a5324bb5d23629::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not reconstruction_matches",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::raise::7f7e7fd14c24037a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Lambda is required to check low-rank reconstruction in the preconditioned geometry')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::numerical_premise_call::e4c5a2b0ecf6bd89::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_factor_projection_certificate(value, factors, lambda_matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::decision_predicate::175084e8c709bf3d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not certificate.valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::raise::ec68508f7bced457::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(certificate.reason)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::compare::de7dc758f1e0c805::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::decision_predicate::de7dc758f1e0c805::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._algebraic_rank_bound::compare::ace6c5f66a2b5554::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value != 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::compare::aeaea49a92e5294d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::decision_predicate::96853989cbd0d8ca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim != 2 or value.shape[0] != value.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::boolean_atom::aeaea49a92e5294d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::compare::66aeee4dac32e967::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.shape[0] != value.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::boolean_atom::66aeee4dac32e967::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.shape[0] != value.shape[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::decision_predicate::bfdbc42f821a1db3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::decision_predicate::cbf16c01011d3c14::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(value, transpose)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::predicate_call_atom::cbf16c01011d3c14::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(value, transpose)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::decision_predicate::fc49e3ff31db631b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "True",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::compare::d165a08a70bb226f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "scale != 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::decision_predicate::b3f16de472f02a1d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.all(difference <= tolerance))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::predicate_call_atom::b3f16de472f02a1d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.all(difference <= tolerance))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::predicate_call_atom::5810b5e15084c7f9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(difference <= tolerance)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::compare::c3355ac3d163349a::0",
        CandidateClassification.NUMERICAL_GATE,
        "difference <= tolerance",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::compare::de7dc758f1e0c805::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::decision_predicate::de7dc758f1e0c805::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::decision_predicate::8b0c93739c0c0f85::0",
        CandidateClassification.NUMERICAL_GATE,
        "bool(np.all(value > 0.0))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::8b0c93739c0c0f85::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.all(value > 0.0))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::f0be7ba5f8b8712e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(value > 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::compare::ede5c27a1b4dedf3::0",
        CandidateClassification.NUMERICAL_GATE,
        "value > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::decision_predicate::f0ef79aec301dc19::0",
        CandidateClassification.NUMERICAL_GATE,
        "not _is_symmetric(value, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::489cb72b71678a88::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_symmetric(value, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::decision_predicate::bfdbc42f821a1db3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::decision_predicate::bfdbc42f821a1db3::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::decision_predicate::88cc28e971f2b8ac::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.all(np.isfinite(eigenvalues)) and np.all(eigenvalues > 0.0))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::88cc28e971f2b8ac::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.all(np.isfinite(eigenvalues)) and np.all(eigenvalues > 0.0))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::decision_predicate::469f1eb6fda2ed51::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(eigenvalues)) and np.all(eigenvalues > 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::boolean_atom::ce0ccc48257ad825::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(eigenvalues))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::ce0ccc48257ad825::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(eigenvalues))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::6ccd49715b3f8858::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(eigenvalues)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::finite_predicate::6ccd49715b3f8858::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(eigenvalues)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::boolean_atom::f9c84ba9f4f55168::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(eigenvalues > 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::f9c84ba9f4f55168::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(eigenvalues > 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::compare::6873fea2c5a9a753::0",
        CandidateClassification.NUMERICAL_GATE,
        "eigenvalues > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::compare::de7dc758f1e0c805::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::decision_predicate::de7dc758f1e0c805::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::compare::bb452dd93c9c2cf2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "smallest == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::decision_predicate::bb452dd93c9c2cf2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "smallest == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::decision_predicate::555bb58ad5c06f76::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.issubdtype(value.dtype, np.inexact)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::predicate_call_atom::8ca616af2dc54785::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.isfinite(condition) and condition < ceiling)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::decision_predicate::3e9d8ac524caed2b::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(condition) and condition < ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::boolean_atom::fc2f1b3e6eabcfd1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(condition)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::predicate_call_atom::fc2f1b3e6eabcfd1::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(condition)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::finite_predicate::fc2f1b3e6eabcfd1::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(condition)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::compare::5c3d08ac757d1ce7::0",
        CandidateClassification.NUMERICAL_GATE,
        "condition < ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::boolean_atom::5c3d08ac757d1ce7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "condition < ceiling",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._require_resolved_dense_condition::compare::de7dc758f1e0c805::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._require_resolved_dense_condition::decision_predicate::de7dc758f1e0c805::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._require_resolved_dense_condition::numerical_premise_call::6adf48d96d85e151::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_condition_certificate(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._require_resolved_dense_condition::decision_predicate::42b9d50503404855::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._require_resolved_dense_condition::raise::3061909c2324f603::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{method} condition {condition:.8g} is not below the strict dtype ceiling {ceiling:.8g}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._x_matrix::compare::230065df746b0dd9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_matrix.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._x_matrix::decision_predicate::230065df746b0dd9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_matrix.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::decision_predicate::2b61f5f86861b6c3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not _is_positive_definite(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::predicate_call_atom::2f5d27d8275ca569::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_positive_definite(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::raise::3f96309c5446e1c0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Lambda must be symmetric positive definite')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::compare::ea5c6a47cd433316::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::decision_predicate::ea5c6a47cd433316::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lam.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::compare::411d89f67a1cb014::0",
        CandidateClassification.NUMERICAL_GATE,
        "maximum < float(np.finfo(lam.dtype).tiny)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::decision_predicate::411d89f67a1cb014::0",
        CandidateClassification.NUMERICAL_GATE,
        "maximum < float(np.finfo(lam.dtype).tiny)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::linalg_exception_premise::35eba272dc25dcee::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "try:\n    with np.errstate(over='raise', divide='raise', invalid='raise'):\n        factor = np.linalg.cholesky(scaled)\n        factor_logdet = 2.0 * math.fsum((float(value) for value in np.log(np.diag(factor))))\n    scale_correction = -float(lam.shape[0] * scale_shift) * math.log(2.0)\n    total = math.fsum((factor_logdet, scale_correction))\nexcept (FloatingPointError, np.linalg.LinAlgError, OverflowError, TypeError, ValueError) as error:\n    raise ValueError('Lambda Cholesky could not be evaluated with finite supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::linalg_call_atom::48f9fc1f91ac5d93::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.linalg.cholesky(scaled)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::raise::1fe22e8a10e5b8f8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Lambda Cholesky could not be evaluated with finite supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::decision_predicate::f517e52fd3adde59::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not math.isfinite(total)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::predicate_call_atom::5662962bf2f78843::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(total)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::finite_predicate::5662962bf2f78843::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(total)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::raise::5b8f42da126a0aa4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Lambda Cholesky could not be evaluated with finite supported arithmetic')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_stability::numerical_premise_call::9f18757e939336dc::0",
        CandidateClassification.NUMERICAL_GATE,
        "spectral_radius(lambda_matrix, perturbation)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_stability::compare::87baab435b8b4904::0",
        CandidateClassification.NUMERICAL_GATE,
        "rho <= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::2b61f5f86861b6c3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not _is_positive_definite(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::predicate_call_atom::2f5d27d8275ca569::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_positive_definite(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::raise::3f96309c5446e1c0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Lambda must be symmetric positive definite')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::numerical_premise_call::4025d304e4487b7c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_two_sum_error(lam, perturb)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::e881b5504ef7e1fc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not _is_positive_definite(sigma)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::predicate_call_atom::43d7c2f9eca8f350::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_positive_definite(sigma)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::raise::443abb82a648012a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Sigma must be symmetric positive definite')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::compare::469cf6ad6cae420a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::469cf6ad6cae420a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::13f99d2de589205e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_factor_projection_certificate(perturb, factors, lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::numerical_premise_call::13f99d2de589205e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_factor_projection_certificate(perturb, factors, lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::175084e8c709bf3d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not certificate.valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::raise::ec68508f7bced457::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(certificate.reason)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::fef369bab0c6f47c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.result_type(lam.dtype, factors.left.dtype, factors.right.dtype)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::compare::67ad7eeeeef81d69::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "work_dtype.type not in (np.float32, np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::67ad7eeeeef81d69::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "work_dtype.type not in (np.float32, np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::c46b9b6f06697401::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.dtype(np.float64)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::1567314e48eb7a53::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.asarray(lam, dtype=work_dtype)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::ad0346d50e2fc12b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.diag(work_lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::feb2d399811644b0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(work_lam, np.diag(diagonal))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::predicate_call_atom::feb2d399811644b0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(work_lam, np.diag(diagonal))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::b56e5e39c83bbabb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_is_diagonal",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::numerical_premise_call::5009547ecebadfa6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet(work_lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::54bad58f7809c38a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.diag(certificate.reduced)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::4c89b2a0c0c67f0f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(certificate.reduced, np.diag(reduced_diagonal))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::predicate_call_atom::4c89b2a0c0c67f0f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(certificate.reduced, np.diag(reduced_diagonal))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::246fd12c2d420b21::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reduced_is_diagonal",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::compare::19707041cdd63cab::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.reduced_q is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::9efb04c0ba628bcc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.reduced_q is None or certificate.reduced_r is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::boolean_atom::19707041cdd63cab::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.reduced_q is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::compare::e549234af47c91bb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.reduced_r is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::boolean_atom::e549234af47c91bb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.reduced_r is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::raise::be1044287921c702::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('the certified reduced determinant-lemma QR payload is missing')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::raise::3679493302ec93ce::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('the reduced determinant-lemma matrix could not be evaluated with finite arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::75f7af2780f9daf3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not (np.all(np.isfinite(certificate.reduced)) and sign > 0.0 and np.isfinite(log_absolute_determinant))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::feba419c8af8f88e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(certificate.reduced)) and sign > 0.0 and np.isfinite(log_absolute_determinant)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::boolean_atom::0bc497daba841835::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(certificate.reduced))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::predicate_call_atom::0bc497daba841835::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(certificate.reduced))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::predicate_call_atom::ddbd39696df8ca9a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(certificate.reduced)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::finite_predicate::ddbd39696df8ca9a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(certificate.reduced)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::compare::a6915774210e08d6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sign > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::boolean_atom::a6915774210e08d6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sign > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::boolean_atom::65859fa6dd4fda20::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(log_absolute_determinant)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::predicate_call_atom::65859fa6dd4fda20::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(log_absolute_determinant)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::finite_predicate::65859fa6dd4fda20::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(log_absolute_determinant)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::raise::94245f821b5dff4b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('the reduced determinant-lemma matrix must be finite and have a positive determinant')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::545f07951e36d507::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "require_finite_stability",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::11d4f5fe8aeca2b4::0",
        CandidateClassification.NUMERICAL_GATE,
        "not stable",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::raise::970d8648f87cbb45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'finite e-polynomial stability cannot certify an expansive spectrum at degree {termination}: measured rho={rho:.8g}. Fall through to a stable exact rung.')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::compare::4de160e7956c5d45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::4de160e7956c5d45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::2775f7451315b08f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_require_resolved_dense_condition(sigma, 'finite e-polynomial')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.low_rank_logdet::numerical_premise_call::92cb69d5d8455220::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_algebraic_rank_bound(perturb, factors, lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.finite_perturbation_logdet::numerical_premise_call::92cb69d5d8455220::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_algebraic_rank_bound(perturb, factors, lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::compare::cbe13cb99f8f4f4e::0",
        CandidateClassification.NUMERICAL_GATE,
        "block_size < 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::decision_predicate::5323d72464a08437::0",
        CandidateClassification.NUMERICAL_GATE,
        "block_size < 1 or n % block_size",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::boolean_atom::cbe13cb99f8f4f4e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "block_size < 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::boolean_atom::bc6a90f435a30f14::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "n % block_size",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::decision_predicate::bfdbc42f821a1db3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::compare::00260021c208fb10::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "abs(row - column) <= 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::decision_predicate::00260021c208fb10::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "abs(row - column) <= 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::decision_predicate::5bc915df4c4a49d1::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.any(piece != 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::predicate_call_atom::5bc915df4c4a49d1::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.any(piece != 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::compare::3b7e7b0a86cea352::0",
        CandidateClassification.NUMERICAL_GATE,
        "piece != 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::decision_predicate::bfdbc42f821a1db3::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::decision_predicate::fc49e3ff31db631b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "True",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::decision_predicate::0546abfc6caa1952::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.all(np.isfinite(value))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::predicate_call_atom::64f310326efef784::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(value))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::predicate_call_atom::6149324691a26641::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::finite_predicate::6149324691a26641::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::raise::8e0426582aacb7dc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{operation} did not produce a finite matrix')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::decision_predicate::54455d18328f13ed::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(value, value.T)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::predicate_call_atom::54455d18328f13ed::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(value, value.T)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::compare::28c43e73cb8cba8f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.signbit(value) == np.signbit(transpose)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::compare::77a7b4bfbc80a9d9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "absolute_value > maximum - absolute_transpose",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::raise::54c353776abc70b6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{operation} could not form a finite symmetric roundoff representative') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::decision_predicate::31d827547ec6f5ec::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.all(np.isfinite(representative))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::predicate_call_atom::d0afd14c40ab35fa::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(representative))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::predicate_call_atom::8df126eb2e95924e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(representative)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::finite_predicate::8df126eb2e95924e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(representative)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._symmetric_roundoff_representative::raise::2a9947559860a621::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{operation} could not form a finite symmetric roundoff representative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::compare::cca48e39ae5072b2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "coefficient_scale == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::decision_predicate::cca48e39ae5072b2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "coefficient_scale == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::raise::817b6e9237bf4f18::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{operation} has a singular all-zero Schur pivot')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::compare::af960f7af7aee0e0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "absolute_matrix != 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::compare::b46bcad19c381db2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right_hand_side != 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::decision_predicate::60a15d1331f7a620::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.frexp(float(np.finfo(matrix.dtype).max))[1]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::decision_predicate::3ce4ee008fb10372::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "largest_exponent - int(np.max(exponents))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::decision_predicate::2457d37c20371e59::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "min(max(0, -coefficient_exponent), upper_shift)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::clamp_selector::2457d37c20371e59::0",
        CandidateClassification.STATIC_SELECTOR,
        "min(max(0, -coefficient_exponent), upper_shift)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::clamp_selector::1ad49da0c13db108::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(0, -coefficient_exponent)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::decision_predicate::ada6aa8fc53f8c14::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.ldexp(matrix, shift)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::decision_predicate::3a2d40d4f7abb18f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.ldexp(right_hand_side, shift)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::raise::9bf056b8ff479723::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{operation} could not be evaluated with finite supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::decision_predicate::f24ed8c02acc7c9b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.array_equal(np.ldexp(scaled_matrix, -shift), matrix) and np.array_equal(np.ldexp(scaled_right_hand_side, -shift), right_hand_side))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::predicate_call_atom::f24ed8c02acc7c9b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.array_equal(np.ldexp(scaled_matrix, -shift), matrix) and np.array_equal(np.ldexp(scaled_right_hand_side, -shift), right_hand_side))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::decision_predicate::4a3f9729486664ad::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(np.ldexp(scaled_matrix, -shift), matrix) and np.array_equal(np.ldexp(scaled_right_hand_side, -shift), right_hand_side)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::boolean_atom::8e189d88bf72bbe3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(np.ldexp(scaled_matrix, -shift), matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::predicate_call_atom::8e189d88bf72bbe3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(np.ldexp(scaled_matrix, -shift), matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::boolean_atom::a33fe3ffd7727d82::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(np.ldexp(scaled_right_hand_side, -shift), right_hand_side)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::predicate_call_atom::a33fe3ffd7727d82::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(np.ldexp(scaled_right_hand_side, -shift), right_hand_side)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::decision_predicate::38c37331da6e3b35::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not (inputs_preserved and np.all(np.isfinite(scaled_matrix)) and np.all(np.isfinite(scaled_right_hand_side)) and np.all(np.isfinite(solution)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::decision_predicate::742481c0a91952c4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "inputs_preserved and np.all(np.isfinite(scaled_matrix)) and np.all(np.isfinite(scaled_right_hand_side)) and np.all(np.isfinite(solution))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::boolean_atom::febe1478137245f4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "inputs_preserved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::boolean_atom::83d4eb0bdda2daaf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(scaled_matrix))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::predicate_call_atom::83d4eb0bdda2daaf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(scaled_matrix))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::predicate_call_atom::6244063a2e3b6641::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(scaled_matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::finite_predicate::6244063a2e3b6641::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(scaled_matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::boolean_atom::1f9affefe224c725::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(scaled_right_hand_side))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::predicate_call_atom::1f9affefe224c725::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(scaled_right_hand_side))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::predicate_call_atom::e56a8b42b91d97ba::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(scaled_right_hand_side)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::finite_predicate::e56a8b42b91d97ba::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(scaled_right_hand_side)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::boolean_atom::37762ca3afbeca85::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(solution))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::predicate_call_atom::37762ca3afbeca85::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(solution))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::predicate_call_atom::d13c87eb87ae1391::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(solution)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::finite_predicate::d13c87eb87ae1391::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(solution)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_of_two_scaled_solve::raise::8ac133e440d092d1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{operation} could not be evaluated with finite supported arithmetic')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::compare::b905442dccd51502::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "maximum == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::decision_predicate::b905442dccd51502::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "maximum == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::raise::9f92d1912b3f3ba5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{operation} input has no nonzero scale')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::compare::fce21e9fc08cd84e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "shift == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::decision_predicate::fce21e9fc08cd84e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "shift == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::compare::79c6781065a65239::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "shift < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::decision_predicate::79c6781065a65239::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "shift < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::raise::d83fe88ad36d2f27::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{operation} input could not be scaled with finite exact arithmetic') from None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::decision_predicate::3b862494ea40bab5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not (np.all(np.isfinite(scaled)) and np.array_equal(restored, matrix))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::decision_predicate::a294baf178737504::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(scaled)) and np.array_equal(restored, matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::boolean_atom::97041e8c804e2dd3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(scaled))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::predicate_call_atom::97041e8c804e2dd3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(scaled))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::predicate_call_atom::b241ba29b74c3708::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(scaled)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::finite_predicate::b241ba29b74c3708::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(scaled)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::boolean_atom::b6abb2edfa930de6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(restored, matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::predicate_call_atom::b6abb2edfa930de6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(restored, matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::compare::79c6781065a65239::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "shift < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::decision_predicate::79c6781065a65239::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "shift < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._exact_power_of_two_scale::raise::60c9db20e1bea684::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{operation} input could not be scaled with finite exact arithmetic')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::decision_predicate::05d181e4f7161009::0",
        CandidateClassification.NUMERICAL_GATE,
        "not _is_block_chain(dense, block_size, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::predicate_call_atom::038ec51d46ad9e6e::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_block_chain(dense, block_size, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::raise::7fcf789928c454c6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'matrix is not a block chain with block_size={block_size}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::decision_predicate::da99982e585a6a0c::0",
        CandidateClassification.NUMERICAL_GATE,
        "not _is_symmetric(dense, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::predicate_call_atom::59cf226300ca4fe8::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_symmetric(dense, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::raise::5b3edd67a7cbe490::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('a block chain logdet requires a symmetric matrix')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::decision_predicate::fd1f83756118c8ce::0",
        CandidateClassification.NUMERICAL_GATE,
        "not _is_positive_definite(dense)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::predicate_call_atom::b838824f5fcd600a::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_positive_definite(dense)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::raise::d1548b1efde5fc08::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('a block chain logdet requires a positive definite matrix')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::decision_predicate::30d95146f3bde552::0",
        CandidateClassification.NUMERICAL_GATE,
        "_require_resolved_dense_condition(dense, 'block-LDL')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::numerical_premise_call::7ab2a9ded33095ce::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet(schur)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::raise::fa1fe9055d9b8c94::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'block-LDL Schur update {index} could not be evaluated with finite supported arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::numerical_premise_call::7ab2a9ded33095ce::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet(schur)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::raise::1f907d6156f879ec::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'block-LDL Schur pivot {index} is not a resolved symmetric positive-definite matrix') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::raise::df0fb425a2f4dadf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('block-LDL pivot log-determinants do not have a finite sum') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::decision_predicate::f517e52fd3adde59::0",
        CandidateClassification.NUMERICAL_GATE,
        "not math.isfinite(total)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::predicate_call_atom::5662962bf2f78843::0",
        CandidateClassification.NUMERICAL_GATE,
        "math.isfinite(total)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::finite_predicate::5662962bf2f78843::0",
        CandidateClassification.NUMERICAL_GATE,
        "math.isfinite(total)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::raise::916c15401224f64a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('block-LDL pivot log-determinants do not have a finite sum')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_diagonal::decision_predicate::c2297a7650c349e2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.array_equal(matrix, np.diag(np.diag(matrix))))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_diagonal::predicate_call_atom::c2297a7650c349e2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.array_equal(matrix, np.diag(np.diag(matrix))))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_diagonal::predicate_call_atom::1df7bc50a873a119::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(matrix, np.diag(np.diag(matrix)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_circulant::decision_predicate::ba2a20ad35ddd4b9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.array_equal(matrix, expected))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_circulant::predicate_call_atom::ba2a20ad35ddd4b9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.array_equal(matrix, expected))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_circulant::predicate_call_atom::eeb4af8582c3e1f5::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(matrix, expected)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._circulant_eigenvalues::decision_predicate::b71014a69ca71b4f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.iscomplexobj(eigenvalues) or np.any(eigenvalues <= 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._circulant_eigenvalues::boolean_atom::75993d11b8210b6a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.iscomplexobj(eigenvalues)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._circulant_eigenvalues::boolean_atom::21dfc06b9583d9ec::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.any(eigenvalues <= 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._circulant_eigenvalues::predicate_call_atom::21dfc06b9583d9ec::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.any(eigenvalues <= 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._circulant_eigenvalues::compare::4ac2ef8d233a8860::0",
        CandidateClassification.NUMERICAL_GATE,
        "eigenvalues <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._circulant_eigenvalues::raise::6ce00499a6ef8651::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('circulant matrix must have a real positive spectrum')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_toeplitz::decision_predicate::c95a6d4c2a63a99f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.all(diagonal == diagonal[0])",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_toeplitz::predicate_call_atom::5bd1473e70b28eee::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(diagonal == diagonal[0])",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_toeplitz::compare::164edfdc38712d7d::0",
        CandidateClassification.NUMERICAL_GATE,
        "diagonal == diagonal[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_toeplitz::decision_predicate::bfdbc42f821a1db3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_toeplitz::decision_predicate::fc49e3ff31db631b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "True",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::compare::12c8c34ea5299bb0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "dense.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::12c8c34ea5299bb0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "dense.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::compare::52a5bd1d310b8030::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind != 'diagonal'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::52a5bd1d310b8030::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind != 'diagonal'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::raise::fff0ab3edb0cc5f8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'a diagonal-vector input is not {kind}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::numerical_premise_call::59b50122a046b5d7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet(dense)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::compare::6ea984843d01ad63::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'diagonal'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::6ea984843d01ad63::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'diagonal'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::8e2690bfc7a4492d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not _is_diagonal(dense, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::predicate_call_atom::8168ed65b7e52c30::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_diagonal(dense, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::raise::b6658cbcdc764dbc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('matrix is not diagonal')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::numerical_premise_call::1af07c48b1e713b7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet(np.diag(dense))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::compare::c3fc04ca2eedeef5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'circulant'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::c3fc04ca2eedeef5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'circulant'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::8800dae2660546be::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not _is_circulant(dense, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::predicate_call_atom::055894230bb0efd8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_circulant(dense, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::raise::9568cd71a729cb3d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('matrix is not circulant')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::76443e2eb97937a0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_require_resolved_dense_condition(dense, 'circulant')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::compare::6f2d74dde551f717::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'toeplitz'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::6f2d74dde551f717::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'toeplitz'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::be97beedf342ee23::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not _is_toeplitz(dense, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::predicate_call_atom::9f4a15a5a3ba970f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_toeplitz(dense, rtol=rtol, atol=atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::raise::02d53a895aac2ddd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('matrix is not Toeplitz')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::ab0ce01e342a98ee::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_require_resolved_dense_condition(dense, 'Toeplitz')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::compare::2035d39280ab03ca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'kronecker'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::2035d39280ab03ca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'kronecker'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::compare::7305d7fb4c1ae781::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "structure is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::7305d7fb4c1ae781::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "structure is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::raise::dd0c6dd25f741ba0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('kronecker evaluation needs factors to verify')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::compare::43a7e748e57b393e::0",
        CandidateClassification.NUMERICAL_GATE,
        "reconstructed.shape != dense.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::ecd14e44b0d8b37f::0",
        CandidateClassification.NUMERICAL_GATE,
        "reconstructed.shape != dense.shape or not np.array_equal(reconstructed, dense)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::boolean_atom::43a7e748e57b393e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reconstructed.shape != dense.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::boolean_atom::e0433a7dfbf76968::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.array_equal(reconstructed, dense)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::predicate_call_atom::1bb392e6480f6eed::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(reconstructed, dense)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::raise::a0a71d3ba6f5faa6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('Kronecker factors do not reconstruct the supplied matrix')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::ed07693e1c0f2bb5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_require_resolved_dense_condition(dense, 'Kronecker')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::raise::cf6a2a1f5f8ed766::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'unsupported structure kind {kind!r}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.dense_cholesky_logdet::numerical_premise_call::e6d96f533059c407::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet(matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::raise::9c02c094bcd1faf2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(failure) from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::decision_predicate::454612947ebca50f::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.all(np.isfinite(x))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::predicate_call_atom::bfc361b4701f87e4::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(x))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::predicate_call_atom::67c48bccbcafebff::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(x)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::finite_predicate::67c48bccbcafebff::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(x)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::raise::23f09c8490aad598::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(failure)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::compare::4de160e7956c5d45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::decision_predicate::4de160e7956c5d45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::raise::9c02c094bcd1faf2::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(failure) from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::decision_predicate::599684cb5a2bcc16::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.all(np.isfinite(eigenvalues))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::predicate_call_atom::ce0ccc48257ad825::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(eigenvalues))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::predicate_call_atom::6ccd49715b3f8858::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(eigenvalues)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::finite_predicate::6ccd49715b3f8858::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(eigenvalues)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::raise::23f09c8490aad598::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(failure)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::numerical_premise_call::9f18757e939336dc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "spectral_radius(lambda_matrix, perturbation)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::compare::ef4381c450597cfd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certified_rho is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::decision_predicate::ef4381c450597cfd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certified_rho is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::decision_predicate::19690c86ac3c4081::0",
        CandidateClassification.NUMERICAL_GATE,
        "not actual_rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::compare::23509a12d46634b8::0",
        CandidateClassification.NUMERICAL_GATE,
        "actual_rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::raise::9175abf784250e2e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'measured rho={actual_rho} does not satisfy strict rho < 1')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::decision_predicate::84b35b0be0b6b1d7::0",
        CandidateClassification.NUMERICAL_GATE,
        "not 0.0 <= certificate < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::compare::2ff2fddab1a04ef2::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 <= certificate < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::raise::06880d87c76ead4a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'trace-log convergence requires certified rho < 1; got {certificate}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::compare::0e23a71d8847891c::0",
        CandidateClassification.NUMERICAL_GATE,
        "actual_rho > certificate",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::decision_predicate::0e23a71d8847891c::0",
        CandidateClassification.NUMERICAL_GATE,
        "actual_rho > certificate",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::raise::a50ccf99fa0dca96::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'rho certificate {certificate} understates measured rho {actual_rho}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._computed_power_traces::compare::4de160e7956c5d45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._computed_power_traces::decision_predicate::4de160e7956c5d45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._computed_power_traces::compare::4de160e7956c5d45::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._computed_power_traces::decision_predicate::4de160e7956c5d45::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._computed_power_traces::compare::4de160e7956c5d45::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._computed_power_traces::decision_predicate::4de160e7956c5d45::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::compare::7f499b566536192a::0",
        CandidateClassification.NUMERICAL_GATE,
        "order < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::decision_predicate::6bcb5682fc2262b5::0",
        CandidateClassification.NUMERICAL_GATE,
        "order < 0 or len(traces) < order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::boolean_atom::7f499b566536192a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "order < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::compare::2ee9c826b03cae0b::0",
        CandidateClassification.NUMERICAL_GATE,
        "len(traces) < order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::boolean_atom::2ee9c826b03cae0b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "len(traces) < order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::decision_predicate::bfdbc42f821a1db3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::decision_predicate::275e02f83dbffe52::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.all(np.isfinite(supplied)) and np.array_equal(supplied, derived))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::predicate_call_atom::275e02f83dbffe52::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.all(np.isfinite(supplied)) and np.array_equal(supplied, derived))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::decision_predicate::24c12791d2599c57::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(supplied)) and np.array_equal(supplied, derived)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::boolean_atom::10e304f715b34f36::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.all(np.isfinite(supplied))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::predicate_call_atom::10e304f715b34f36::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(supplied))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::predicate_call_atom::7664e7df114ecb98::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(supplied)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::finite_predicate::7664e7df114ecb98::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(supplied)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::boolean_atom::d1ce1fecca04a454::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(supplied, derived)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::predicate_call_atom::d1ce1fecca04a454::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(supplied, derived)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.trace_log_tail_bound::decision_predicate::96a33b68a27091f1::0",
        CandidateClassification.NUMERICAL_GATE,
        "not 0.0 <= rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.trace_log_tail_bound::compare::6f6bf7e28f2b4e29::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 <= rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.trace_log_tail_bound::raise::2d2599128088fe0b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'trace-log convergence requires rho < 1; got {rho}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.trace_log_tail_bound::compare::7f499b566536192a::0",
        CandidateClassification.NUMERICAL_GATE,
        "order < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.trace_log_tail_bound::decision_predicate::7f499b566536192a::0",
        CandidateClassification.NUMERICAL_GATE,
        "order < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.trace_log_tail_bound::raise::44ceae07b2a3223e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('trace-log order must be non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.whole_trace_log_tail_bound::compare::cb1557568a2a2887::0",
        CandidateClassification.NUMERICAL_GATE,
        "multiplicity < 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.whole_trace_log_tail_bound::decision_predicate::cb1557568a2a2887::0",
        CandidateClassification.NUMERICAL_GATE,
        "multiplicity < 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.whole_trace_log_tail_bound::raise::89e325f6a9b763b3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('multiplicity must be positive')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::decision_predicate::db7fa083851799e3::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(tolerance) or tolerance <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::boolean_atom::da15013e08293446::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(tolerance)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::predicate_call_atom::181c8e652b10fae7::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(tolerance)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::finite_predicate::181c8e652b10fae7::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(tolerance)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::compare::d42259ba0e4e85a6::0",
        CandidateClassification.NUMERICAL_GATE,
        "tolerance <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::boolean_atom::d42259ba0e4e85a6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "tolerance <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::raise::ac97a7b8cd1e6ff1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('tolerance must be finite and positive')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::compare::be8bebac0fc5f981::0",
        CandidateClassification.NUMERICAL_GATE,
        "whole_trace_log_tail_bound(rho, order, multiplicity) > tolerance",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::decision_predicate::be8bebac0fc5f981::0",
        CandidateClassification.NUMERICAL_GATE,
        "whole_trace_log_tail_bound(rho, order, multiplicity) > tolerance",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::decision_predicate::1b656269a2014fdb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_validate_strict_rho(lam, perturb, rho)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::compare::fd39062c1e2458c5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "exact_power_traces is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::decision_predicate::fd39062c1e2458c5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "exact_power_traces is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::raise::b3b7398de64f62ad::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('rung 6 requires deterministic exact power traces; one generic matvec cannot provide Tr(X**r)')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::compare::7f499b566536192a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "order < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::decision_predicate::6bcb5682fc2262b5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "order < 0 or len(traces) < order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::boolean_atom::7f499b566536192a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "order < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::compare::2ee9c826b03cae0b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "len(traces) < order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::boolean_atom::2ee9c826b03cae0b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "len(traces) < order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::raise::e42895e0821d05e2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('exact power traces must contain every requested order')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::decision_predicate::63d0120a57104d34::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not _power_traces_match(lam, perturb, traces, order)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::numerical_premise_call::1ca27509710c3312::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_power_traces_match(lam, perturb, traces, order)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::raise::d677fbe31760f6c5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('supplied exact power traces do not match traces derived from X')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.truncated_trace_logdet::numerical_premise_call::ef6686c3eb65e8d5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::compare::a463584319db77e0::0",
        CandidateClassification.NUMERICAL_GATE,
        "type(probes) is not FrozenProbes",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::decision_predicate::a463584319db77e0::0",
        CandidateClassification.NUMERICAL_GATE,
        "type(probes) is not FrozenProbes",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::raise::abf5aa302d250606::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise TypeError('frozen_probes must be an exact FrozenProbes bytes-backed instance')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::decision_predicate::1b656269a2014fdb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_validate_strict_rho(lam, perturb, rho)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::compare::5fe9ad3578e110cc::0",
        CandidateClassification.NUMERICAL_GATE,
        "vectors.shape[1] != _n(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::decision_predicate::5fe9ad3578e110cc::0",
        CandidateClassification.NUMERICAL_GATE,
        "vectors.shape[1] != _n(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::raise::d9d46a7d7f8277b5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('frozen probe width must equal the matrix dimension')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::compare::7f499b566536192a::0",
        CandidateClassification.NUMERICAL_GATE,
        "order < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::decision_predicate::7f499b566536192a::0",
        CandidateClassification.NUMERICAL_GATE,
        "order < 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::raise::e363f6ec8052729b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('order must be non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::compare::4de160e7956c5d45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::decision_predicate::4de160e7956c5d45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::numerical_premise_call::ef6686c3eb65e8d5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet(lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_eager.py::<module>.resampled_trace_logdet::raise::1f5ac479fb5ea756::0",
        CandidateClassification.POLICY_STATIC_REFUSAL,
        "raise ResamplingRefused('Per-call probe resampling makes the log determinant noisy, breaks HMC reversibility, and is always refused. Supply FrozenProbes instead.')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::compare::3204fd05446ce318::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::decision_predicate::aa5e8dd848a0018b::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma.ndim == 1 or np.array_equal(sigma, sigma.T) or (not _is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::boolean_atom::3204fd05446ce318::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::boolean_atom::3916090d68cfe569::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(sigma, sigma.T)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::predicate_call_atom::3916090d68cfe569::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(sigma, sigma.T)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::boolean_atom::b3c78a1f0470782d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not _is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::predicate_call_atom::5092fdf71a6d9964::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::3204fd05446ce318::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::3204fd05446ce318::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::89bf95eccb3933be::0",
        CandidateClassification.NUMERICAL_GATE,
        "bool(np.all(sigma > 0.0))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::d003bc8bdc0751d7::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(sigma > 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::d1a71af8a65a9965::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::cf19d788dfb142fc::0",
        CandidateClassification.NUMERICAL_GATE,
        "kind is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::e4f1323628f87cb8::0",
        CandidateClassification.NUMERICAL_GATE,
        "kind is None and _is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::boolean_atom::cf19d788dfb142fc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::boolean_atom::3b501ceeac5159d6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::3b501ceeac5159d6::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::6ea984843d01ad63::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'diagonal'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::6ea984843d01ad63::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'diagonal'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::3b501ceeac5159d6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::3b501ceeac5159d6::1",
        CandidateClassification.NUMERICAL_GATE,
        "_is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::e0d28a5085f237c2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::c3fc04ca2eedeef5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'circulant'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::c3fc04ca2eedeef5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'circulant'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::7ead0b7ae54d3a4a::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_circulant(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::7ead0b7ae54d3a4a::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_circulant(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::e0d28a5085f237c2::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::e0d28a5085f237c2::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::6f2d74dde551f717::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'toeplitz'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::6f2d74dde551f717::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'toeplitz'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::6a0aad293d04c15d::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_toeplitz(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::6a0aad293d04c15d::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_toeplitz(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::e0d28a5085f237c2::3",
        CandidateClassification.ORDINARY_VALIDATION,
        "valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::2035d39280ab03ca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'kronecker'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::2035d39280ab03ca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "kind == 'kronecker'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::eeee57a0fe7dd331::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.structure is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::eeee57a0fe7dd331::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.structure is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::eb776e7e35594b34::0",
        CandidateClassification.NUMERICAL_GATE,
        "all((_is_positive_definite(factor) for factor in problem.structure.factors))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::75f6a0da8f1923cb::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_positive_definite(factor)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::d815c74c8c74ed82::0",
        CandidateClassification.NUMERICAL_GATE,
        "not factors_spd",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::674c6611419912a0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.structure.factors[0]",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::3f0083b114a6b5f7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.kron(reconstructed, factor)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::8cea1b9d73a001b5::0",
        CandidateClassification.NUMERICAL_GATE,
        "reconstructed.shape == sigma.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::d611227f5fbefbc9::0",
        CandidateClassification.NUMERICAL_GATE,
        "reconstructed.shape == sigma.shape and np.array_equal(reconstructed, sigma)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::boolean_atom::8cea1b9d73a001b5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "reconstructed.shape == sigma.shape",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::boolean_atom::2aaa6008086ab43f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(reconstructed, sigma)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::2aaa6008086ab43f::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.array_equal(reconstructed, sigma)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::50f70b70544a44e7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(valid)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::e0d28a5085f237c2::4",
        CandidateClassification.ORDINARY_VALIDATION,
        "valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::6a8862627553dd32::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "LadderConfig() if config is None else config",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::96087f1978267d74::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "config is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::96087f1978267d74::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "config is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::8fdae9f9716a3854::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.lambda_matrix",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::58055b99e87063f2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.perturbation",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::4025d304e4487b7c::0",
        CandidateClassification.NUMERICAL_GATE,
        "_two_sum_error(lam, perturb)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::bfdbc42f821a1db3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::bfdbc42f821a1db3::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::bfdbc42f821a1db3::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::436dbd53184d8935::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.issubdtype(lam.dtype, np.inexact)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::bfdbc42f821a1db3::3",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::fc49e3ff31db631b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "True",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::3204fd05446ce318::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::afce5741cc3cec3d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1 or bool(np.array_equal(sigma, sigma.T))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::3204fd05446ce318::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::40fcc52becea2e8b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.array_equal(sigma, sigma.T))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::40fcc52becea2e8b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.array_equal(sigma, sigma.T))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::3916090d68cfe569::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(sigma, sigma.T)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::3204fd05446ce318::1",
        CandidateClassification.NUMERICAL_GATE,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::e990d931da5e772e::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma.ndim == 1 or _is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::3204fd05446ce318::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::5092fdf71a6d9964::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::5092fdf71a6d9964::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::fae93f08468cafbe::0",
        CandidateClassification.NUMERICAL_GATE,
        "_sigma_payload(sigma, config)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::7ce742d10265b420::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_symmetric and _is_positive_definite(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::8cb15373fb023923::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma_symmetric",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::ded5826f515f9d49::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_positive_definite(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::ded5826f515f9d49::0",
        CandidateClassification.NUMERICAL_GATE,
        "_is_positive_definite(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::e67c5addca58cd75::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_condition_certificate(symmetric_sigma)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::2f33a7416d448daf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.low_rank_factors is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::2f33a7416d448daf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.low_rank_factors is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::32b77c3c606be73a::0",
        CandidateClassification.NUMERICAL_GATE,
        "_algebraic_rank_bound(perturb)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::fc49e3ff31db631b::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "True",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::fe4f7ab1beecdb99::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_factor_projection_certificate(perturb, problem.low_rank_factors, lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::fe4f7ab1beecdb99::0",
        CandidateClassification.NUMERICAL_GATE,
        "_factor_projection_certificate(perturb, problem.low_rank_factors, lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::bfdbc42f821a1db3::4",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::24dbc67323cd27ce::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factor_certificate.valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::c515480aa003dc6d::0",
        CandidateClassification.NUMERICAL_GATE,
        "rank_evidence_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::a443ff27e05df6b5::0",
        CandidateClassification.NUMERICAL_GATE,
        "spectral_radius(lam, perturb)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::ad0571b6919536d8::0",
        CandidateClassification.NUMERICAL_GATE,
        "not sigma_formation_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::d3068b908ee92dee::0",
        CandidateClassification.NUMERICAL_GATE,
        "spectral_radius(lam, finite_payload_perturbation)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::057b7b726f746fbf::0",
        CandidateClassification.NUMERICAL_GATE,
        "finite_payload_rho_measurement_valid and finite_payload_rho <= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::d9fb4fc3637a9139::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "finite_payload_rho_measurement_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::4a956d9542c5c0c2::0",
        CandidateClassification.NUMERICAL_GATE,
        "finite_payload_rho <= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::4a956d9542c5c0c2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "finite_payload_rho <= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::96e427b5f105da27::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(problem.low_rank_factors is not None and rank_evidence_valid and sigma_formation_valid and sigma_exactly_symmetric)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::14820d49ac03ef64::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.low_rank_factors is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::8f473f76dc3f6936::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.low_rank_factors is not None and rank_evidence_valid and sigma_formation_valid and sigma_exactly_symmetric",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::14820d49ac03ef64::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.low_rank_factors is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::c515480aa003dc6d::0",
        CandidateClassification.NUMERICAL_GATE,
        "rank_evidence_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::48196fb8375388cd::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_formation_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::91367abf12d9ecf4::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_exactly_symmetric",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::802e5f1296309604::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "finite_polynomial_stable or determinant_lemma_payload",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::95e8e34248eca4da::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "finite_polynomial_stable",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::fee2daab0ef0a42e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "determinant_lemma_payload",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::88c73547f0f976e9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma_formation_valid and (sigma.ndim == 1 or condition_resolved)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::48196fb8375388cd::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma_formation_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::3204fd05446ce318::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::384e9ebea3a6e949::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1 or condition_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::3204fd05446ce318::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::6bb5018fea5f3016::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "condition_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::7fbc415aee154691::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_formation_valid and bool(np.array_equal(sigma, lam)) and dense_arithmetic_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::48196fb8375388cd::2",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_formation_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::d9082e7801c0ca71::0",
        CandidateClassification.NUMERICAL_GATE,
        "bool(np.array_equal(sigma, lam))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::d9082e7801c0ca71::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(np.array_equal(sigma, lam))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::0daa42baad4d8b5a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.array_equal(sigma, lam)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::969c43f564c8d1e8::0",
        CandidateClassification.NUMERICAL_GATE,
        "dense_arithmetic_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::88ab78fc74ff69ec::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "perturb.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::efa1c96a45b8a99d::0",
        CandidateClassification.NUMERICAL_GATE,
        "rank_evidence_valid and (compact_diagonal_payload or determinant_lemma_payload) and sigma_spd and (rank <= config.low_rank_max) and (rank <= config.low_rank_fraction * n)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::c515480aa003dc6d::1",
        CandidateClassification.NUMERICAL_GATE,
        "rank_evidence_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::df59227e94deb60c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "compact_diagonal_payload or determinant_lemma_payload",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::772b10fa9b76b884::0",
        CandidateClassification.NUMERICAL_GATE,
        "compact_diagonal_payload",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::fee2daab0ef0a42e::1",
        CandidateClassification.NUMERICAL_GATE,
        "determinant_lemma_payload",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2acb41d6956562c0::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_spd",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::44f3815eb0ce7bcd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rank <= config.low_rank_max",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::44f3815eb0ce7bcd::0",
        CandidateClassification.NUMERICAL_GATE,
        "rank <= config.low_rank_max",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::59f66427dc73ee94::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rank <= config.low_rank_fraction * n",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::59f66427dc73ee94::0",
        CandidateClassification.NUMERICAL_GATE,
        "rank <= config.low_rank_fraction * n",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::ad0571b6919536d8::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "not sigma_formation_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::408ecc0906f78d6a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.chain_block_size is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::14842e61fe52435f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.chain_block_size is None or sigma.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::408ecc0906f78d6a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.chain_block_size is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::715ca161fb578e90::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::715ca161fb578e90::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim != 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::b899eab859197a78::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_block_chain(sigma, problem.chain_block_size, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::b899eab859197a78::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_is_block_chain(sigma, problem.chain_block_size, rtol=config.structure_rtol, atol=config.structure_atol)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::7d56f970127e454a::0",
        CandidateClassification.NUMERICAL_GATE,
        "chain_structure and sigma_spd and condition_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2df45b3b0a16f43b::0",
        CandidateClassification.NUMERICAL_GATE,
        "chain_structure",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2acb41d6956562c0::1",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_spd",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::6bb5018fea5f3016::1",
        CandidateClassification.NUMERICAL_GATE,
        "condition_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::ca97111449f3704c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not chain_structure",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::d8043794f0f969b2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not sigma_symmetric",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::151f00f7d5de0f61::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not sigma_spd",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::d69a1ae0cf7be478::0",
        CandidateClassification.NUMERICAL_GATE,
        "not condition_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::48196fb8375388cd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma_formation_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::bfdbc42f821a1db3::5",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::23fecd9cd417499b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "structured and (not sigma_spd)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2ea8744b9588c9b7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "structured",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::151f00f7d5de0f61::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not sigma_spd",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::bfdbc42f821a1db3::6",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::d8043794f0f969b2::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "not sigma_symmetric",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::aef6e67f06596e6b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "structured and structure_kind != 'diagonal' and (not condition_resolved)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2ea8744b9588c9b7::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "structured",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::f18711b1926dfe54::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "structure_kind != 'diagonal'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::f18711b1926dfe54::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "structure_kind != 'diagonal'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::d69a1ae0cf7be478::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not condition_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::bfdbc42f821a1db3::7",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::f0ba3f1026fc300c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "n <= config.dense_max_n",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::3ca3ab1fd1c3fc27::0",
        CandidateClassification.NUMERICAL_GATE,
        "n <= config.dense_max_n and condition_resolved and sigma_spd",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::f0ba3f1026fc300c::0",
        CandidateClassification.NUMERICAL_GATE,
        "n <= config.dense_max_n",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::6bb5018fea5f3016::2",
        CandidateClassification.NUMERICAL_GATE,
        "condition_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2acb41d6956562c0::2",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_spd",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::5b64ae2cb778a757::0",
        CandidateClassification.NUMERICAL_GATE,
        "n <= config.finite_max_n",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::6d8dda4ea60f3e75::0",
        CandidateClassification.NUMERICAL_GATE,
        "n <= config.finite_max_n or ((compact_diagonal_payload or determinant_lemma_payload) and rank <= config.finite_max_rank)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::5b64ae2cb778a757::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "n <= config.finite_max_n",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::554111cb2ca3295d::0",
        CandidateClassification.NUMERICAL_GATE,
        "(compact_diagonal_payload or determinant_lemma_payload) and rank <= config.finite_max_rank",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::df59227e94deb60c::1",
        CandidateClassification.NUMERICAL_GATE,
        "compact_diagonal_payload or determinant_lemma_payload",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::772b10fa9b76b884::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "compact_diagonal_payload",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::fee2daab0ef0a42e::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "determinant_lemma_payload",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::80d02c27a4b17ad0::0",
        CandidateClassification.NUMERICAL_GATE,
        "rank <= config.finite_max_rank",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::80d02c27a4b17ad0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rank <= config.finite_max_rank",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::ed70d0b90f2a707b::0",
        CandidateClassification.NUMERICAL_GATE,
        "finite_size_qualified and finite_payload_stable and sigma_spd and (determinant_lemma_payload or dense_arithmetic_resolved)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::57439eb4dffa740a::0",
        CandidateClassification.NUMERICAL_GATE,
        "finite_size_qualified",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::cbe2295ff59158de::0",
        CandidateClassification.NUMERICAL_GATE,
        "finite_payload_stable",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2acb41d6956562c0::3",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_spd",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::6a781d45dfcc23af::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "determinant_lemma_payload or dense_arithmetic_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::fee2daab0ef0a42e::3",
        CandidateClassification.NUMERICAL_GATE,
        "determinant_lemma_payload",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::969c43f564c8d1e8::1",
        CandidateClassification.NUMERICAL_GATE,
        "dense_arithmetic_resolved",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::46b79af2a8ab3b62::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.certified_rho is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::46b79af2a8ab3b62::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.certified_rho is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::53ce5b335829ca05::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rho_measurement_valid and actual_rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::355615dbeccb3999::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rho_measurement_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::23509a12d46634b8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "actual_rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::23509a12d46634b8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "actual_rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::c7320981b93b6672::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rho_measurement_valid and actual_rho <= rho",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::355615dbeccb3999::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "rho_measurement_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::25fbd5724c6724f2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "actual_rho <= rho",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::25fbd5724c6724f2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "actual_rho <= rho",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::7151085930139684::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rho_measurement_valid and problem.exact_power_traces is not None and (problem.trace_order is not None) and _power_traces_match(lam, perturb, problem.exact_power_traces, problem.trace_order)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::355615dbeccb3999::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "rho_measurement_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::d9c49a04b7652361::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.exact_power_traces is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::d9c49a04b7652361::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.exact_power_traces is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::c62b0e04d76df049::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.trace_order is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::c62b0e04d76df049::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.trace_order is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::5073c24219af64e1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_power_traces_match(lam, perturb, problem.exact_power_traces, problem.trace_order)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::5073c24219af64e1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_power_traces_match(lam, perturb, problem.exact_power_traces, problem.trace_order)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::7f581995d7238d06::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_formation_valid and traces_verified and measured_rho_converges and rho_covers_input and (0.0 <= rho < 1.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::48196fb8375388cd::3",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_formation_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::bc108dc8b78fe54f::0",
        CandidateClassification.NUMERICAL_GATE,
        "traces_verified",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::5d6aaa0c807a1bf8::0",
        CandidateClassification.NUMERICAL_GATE,
        "measured_rho_converges",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::75bdf97ba8300ea7::0",
        CandidateClassification.NUMERICAL_GATE,
        "rho_covers_input",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::6f6bf7e28f2b4e29::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "0.0 <= rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::6f6bf7e28f2b4e29::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 <= rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::f747be9d50134d03::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.frozen_probes is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::2b59705c911ff576::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.frozen_probes is not None and problem.frozen_probes.values.shape[1] == n",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::f747be9d50134d03::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.frozen_probes is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::01b5f9aec75223c4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.frozen_probes.values.shape[1] == n",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::01b5f9aec75223c4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.frozen_probes.values.shape[1] == n",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::1f75b0ed3ab0d827::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_formation_valid and frozen_width_valid and (problem.trace_order is not None) and (problem.trace_order >= 0) and measured_rho_converges and rho_covers_input and (0.0 <= rho < 1.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::48196fb8375388cd::4",
        CandidateClassification.NUMERICAL_GATE,
        "sigma_formation_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::c087bc09f45f17ee::0",
        CandidateClassification.NUMERICAL_GATE,
        "frozen_width_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::c62b0e04d76df049::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.trace_order is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::c62b0e04d76df049::1",
        CandidateClassification.NUMERICAL_GATE,
        "problem.trace_order is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::5a38bc5f8f0f2d61::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.trace_order >= 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::5a38bc5f8f0f2d61::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.trace_order >= 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::5d6aaa0c807a1bf8::1",
        CandidateClassification.NUMERICAL_GATE,
        "measured_rho_converges",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::75bdf97ba8300ea7::1",
        CandidateClassification.NUMERICAL_GATE,
        "rho_covers_input",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::6f6bf7e28f2b4e29::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "0.0 <= rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::6f6bf7e28f2b4e29::1",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 <= rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::gate_qualifier::fc3273a19cbaa810::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "base",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::fc3273a19cbaa810::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "base",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::gate_qualifier::4388ae6627c8aec4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "low_rank",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::c515480aa003dc6d::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "rank_evidence_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::gate_qualifier::17b6068f79b23f93::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "chain",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::gate_qualifier::2ea8744b9588c9b7::0",
        CandidateClassification.NUMERICAL_GATE,
        "structured",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::gate_qualifier::f38a1a2c5fc65cef::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "dense",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::gate_qualifier::594ffb0aa6631b41::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "finite",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::gate_qualifier::d8ec579729367dc1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "trace",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::d8ec579729367dc1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "trace",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::23615e7190849b10::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not rho_measurement_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::gate_qualifier::3a459345dea9c31e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::3a459345dea9c31e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::23615e7190849b10::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "not rho_measurement_valid",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::policy_literal::bfdbc42f821a1db3::0",
        CandidateClassification.POLICY_STATIC_REFUSAL,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::compare::96087f1978267d74::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "config is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::96087f1978267d74::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "config is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::5e93c8849ae6b944::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not any((verdict.satisfied for verdict in verdicts))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::raise::3da3ad28c76cc0b2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ResamplingRefused('No deterministic log-determinant rung qualified; per-call resampling is HMC-unsafe and refused.', rejected=verdicts)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::numerical_premise_call::126f03455393f019::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_two_sum_error(problem.lambda_matrix, problem.perturbation)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::raise::c28d60afc1454347::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ResamplingRefused(f'No deterministic log-determinant rung qualified because {error}; per-call resampling is HMC-unsafe and refused.', rejected=verdicts) from None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::numerical_premise_call::fae93f08468cafbe::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_sigma_payload(sigma, config)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::b85ecae26de717b8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not verdict.satisfied",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::compare::2cc989f296f9df84::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::2cc989f296f9df84::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::numerical_premise_call::45ae0698f9e638df::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet(problem.lambda_matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::compare::32fbc19c6ffcaa46::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::32fbc19c6ffcaa46::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::compare::591da3ee536806ff::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::591da3ee536806ff::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 2",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::compare::02ed0a3775d3cff6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 3",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::02ed0a3775d3cff6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 3",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::compare::fef60c1438def4c7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 4",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::fef60c1438def4c7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 4",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::compare::a7a06e4df5079ee2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 5",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::a7a06e4df5079ee2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 5",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::eb0dde2619141ffe::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.details['determinant_lemma_payload']",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::compare::469cf6ad6cae420a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::469cf6ad6cae420a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "factors is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::compare::7b4e2dd71f84616f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 6",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::decision_predicate::7b4e2dd71f84616f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.level == 6",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.dispatch_logdet::raise::3da3ad28c76cc0b2::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ResamplingRefused('No deterministic log-determinant rung qualified; per-call resampling is HMC-unsafe and refused.', rejected=verdicts)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::decision_predicate::b26d8d6f4a246b6c::0",
        CandidateClassification.NUMERICAL_GATE,
        "isinstance(value, (bool, np.bool_))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::raise::f7db2e01d8e3b3a6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise TypeError('rho certificate multiplicity must be an integer index, not bool')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::numerical_premise_call::b987512875e7714f::0",
        CandidateClassification.NUMERICAL_GATE,
        "operator.index(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::raise::c94f7ad5bc93b895::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise TypeError('rho certificate multiplicity must be an integer index') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::compare::cb1557568a2a2887::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "multiplicity < 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::decision_predicate::cb1557568a2a2887::0",
        CandidateClassification.NUMERICAL_GATE,
        "multiplicity < 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::raise::6fdcd5bf1744c6b0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('rho certificate multiplicity must be positive')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::compare::c19d43bd613dda15::0",
        CandidateClassification.NUMERICAL_GATE,
        "multiplicity >= _RHO_MULTIPLICITY_LIMIT",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::decision_predicate::c19d43bd613dda15::0",
        CandidateClassification.NUMERICAL_GATE,
        "multiplicity >= _RHO_MULTIPLICITY_LIMIT",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::raise::af73b9e150805782::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('rho certificate multiplicity must satisfy multiplicity * float64 eps < 1')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::ad10c538bb5d448c::0",
        CandidateClassification.NUMERICAL_GATE,
        "not 0.0 <= self.measured_max < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::cfd9ed6ff01db72b::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 <= self.measured_max < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::raise::1afb60fa0bea0aa8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('a rho certificate needs measured_max < 1')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::26baf29e9d74ae01::0",
        CandidateClassification.NUMERICAL_GATE,
        "not 0.0 <= self.certified_rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::ddc2a9a5cd66ebe3::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 <= self.certified_rho < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::raise::0ff3d9e82e8bb9e9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('a rho certificate needs certified_rho < 1')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::1c74be788b92b3e1::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.certified_rho < self.measured_max",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::1c74be788b92b3e1::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.certified_rho < self.measured_max",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::raise::1d134cab3bc0c2fa::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('certified_rho must cover measured_max')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::502e1e1ca5864853::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::6993e0eaa405cc88::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.margin < 0.0 or self.tolerance <= 0.0 or (not 0.0 < self.tail_tolerance < self.tolerance)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::boolean_atom::502e1e1ca5864853::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::08cf8960c456a2ca::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.tolerance <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::boolean_atom::08cf8960c456a2ca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.tolerance <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::boolean_atom::41fc9692e85df8a0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not 0.0 < self.tail_tolerance < self.tolerance",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::d73054a840b6cacf::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 < self.tail_tolerance < self.tolerance",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::raise::467c40df7ecac87c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('rho certificate margin/tolerance/multiplicity are invalid')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::07025b3c333759d1::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.max_abs_lambda_logdet is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::11003df0a5e25fa8::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.max_abs_lambda_logdet is not None and (not np.isfinite(self.max_abs_lambda_logdet) or self.max_abs_lambda_logdet < 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::boolean_atom::07025b3c333759d1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.max_abs_lambda_logdet is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::98328934d03bd7b3::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(self.max_abs_lambda_logdet) or self.max_abs_lambda_logdet < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::boolean_atom::9e7f27b58b3fee70::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(self.max_abs_lambda_logdet)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::predicate_call_atom::d326b31a3a3e51a1::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(self.max_abs_lambda_logdet)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::finite_predicate::d326b31a3a3e51a1::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(self.max_abs_lambda_logdet)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::271377e809b7b124::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.max_abs_lambda_logdet < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::boolean_atom::271377e809b7b124::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.max_abs_lambda_logdet < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::raise::d71e4725f2f09c31::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('lambda-logdet scale bound must be finite and non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::f270ed94716d3f21::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.max_x_operator_norm is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::a0a6c3092291c7f1::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.max_x_operator_norm is not None and (not np.isfinite(self.max_x_operator_norm) or self.max_x_operator_norm < 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::boolean_atom::f270ed94716d3f21::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.max_x_operator_norm is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::88d32009a53226a5::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(self.max_x_operator_norm) or self.max_x_operator_norm < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::boolean_atom::8d7595dfa80e654a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(self.max_x_operator_norm)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::predicate_call_atom::c3c046963e6ed35a::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(self.max_x_operator_norm)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::finite_predicate::c3c046963e6ed35a::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(self.max_x_operator_norm)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::0ac44251ac6cc986::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.max_x_operator_norm < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::boolean_atom::0ac44251ac6cc986::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.max_x_operator_norm < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::raise::70eacb9a3e0b864a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('X operator-norm bound must be finite and non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::5278515c7ba5de86::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.order != expected_order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::5278515c7ba5de86::0",
        CandidateClassification.NUMERICAL_GATE,
        "self.order != expected_order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::raise::4dd163ae187a2aa7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'rho certificate order must be the bound-selected {expected_order}, got {self.order}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.TraceLogPlan.__init__::compare::dbd899e0214ad067::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "token is not _PLAN_TOKEN",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.TraceLogPlan.__init__::decision_predicate::dbd899e0214ad067::0",
        CandidateClassification.POLICY_STATIC_REFUSAL,
        "token is not _PLAN_TOKEN",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.TraceLogPlan.__init__::raise::00331f66fbfa4bc6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise TypeError('TraceLogPlan must be created by make_trace_log_plan')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.TraceLogPlan.__call__::decision_predicate::6d4775268de124d0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_require_runtime_precision(self._runtime_dtype, lambda_logdet_value, exact_power_traces)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.FrozenTraceLogPlan.__init__::compare::dbd899e0214ad067::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "token is not _PLAN_TOKEN",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.FrozenTraceLogPlan.__init__::decision_predicate::dbd899e0214ad067::0",
        CandidateClassification.POLICY_STATIC_REFUSAL,
        "token is not _PLAN_TOKEN",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.FrozenTraceLogPlan.__init__::raise::0ca8cf00f66b3ee1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise TypeError('FrozenTraceLogPlan must be created by make_frozen_trace_log_plan')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.FrozenTraceLogPlan.__call__::decision_predicate::6cc2c135fcf237b6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_require_runtime_precision(self._runtime_dtype, lambda_logdet_value, x_matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::a6a43a5630fe3e1b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "tuple((float(value) for value in measured_rhos))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::57c724e170630987::0",
        CandidateClassification.NUMERICAL_GATE,
        "not values or any((not np.isfinite(value) or value < 0.0 for value in values))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::02429b45018a91c2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not values",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::362cdb8088bd512d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "any((not np.isfinite(value) or value < 0.0 for value in values))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::7b5386c0f5e68697::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(value) or value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::2776f1f61aa53ddd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::6149324691a26641::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::6149324691a26641::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::ceb972d0b12db89c::0",
        CandidateClassification.NUMERICAL_GATE,
        "value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::ceb972d0b12db89c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::raise::18d1138d5f4a14cd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('warmup rho measurements must be non-empty, finite, and non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::83c56a26785d2596::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(margin) or margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::e1424e4bf6f493fc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::e9cbd7a44ba3a956::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::e9cbd7a44ba3a956::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::da1d4844a773cbd5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::da1d4844a773cbd5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::raise::6137273b10c31463::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('rho safety margin must be finite and non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::df7a4a82222ab703::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(tail_fraction) or not 0.0 < tail_fraction < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::0c94ecfd5a8eebac::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(tail_fraction)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::d7756508c0ebfa8c::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(tail_fraction)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::d7756508c0ebfa8c::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(tail_fraction)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::87c7675e0fea6fcd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not 0.0 < tail_fraction < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::6d40d228cfffeb1e::0",
        CandidateClassification.NUMERICAL_GATE,
        "0.0 < tail_fraction < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::raise::bf1b0eb6df7f8c91::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('tail_fraction must lie strictly between zero and one')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::fe9a70b6393b0a8a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "None if lambda_logdets is None else tuple((float(x) for x in lambda_logdets))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::4373b5290fad067c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdets is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::4373b5290fad067c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdets is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::c54d59df26bf8f15::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bases is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::5c34feab94d988ce::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bases is not None and (not bases or any((not np.isfinite(value) for value in bases)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::c54d59df26bf8f15::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bases is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::d52eae775e943fd0::0",
        CandidateClassification.NUMERICAL_GATE,
        "not bases or any((not np.isfinite(value) for value in bases))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::7be74958a2723753::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not bases",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::97d1c12bcefd0340::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "any((not np.isfinite(value) for value in bases))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::6149324691a26641::1",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::6149324691a26641::1",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::raise::9d84d76a8cd1582f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('warmup lambda logdets must be non-empty and finite')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::d619e405863b0f28::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(lambda_logdet_margin) or lambda_logdet_margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::f033cac39ddaa24f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(lambda_logdet_margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::b741736709d56096::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(lambda_logdet_margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::b741736709d56096::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(lambda_logdet_margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::f4f39dc0de81efbf::0",
        CandidateClassification.NUMERICAL_GATE,
        "lambda_logdet_margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::f4f39dc0de81efbf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet_margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::raise::d143bf1300807ca0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('lambda-logdet safety margin must be finite and non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::01d84359be16e08c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "None if x_operator_norms is None else tuple((float(value) for value in x_operator_norms))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::f0df80262bf2b49e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x_operator_norms is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::f0df80262bf2b49e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x_operator_norms is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::d8920b0c982b22be::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "norms is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::b0defd1975adcc6b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "norms is not None and (not norms or any((not np.isfinite(value) or value < 0.0 for value in norms)))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::d8920b0c982b22be::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "norms is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::793fd4e8d52e933e::0",
        CandidateClassification.NUMERICAL_GATE,
        "not norms or any((not np.isfinite(value) or value < 0.0 for value in norms))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::7ce9343023d7d169::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not norms",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::76bca5d6715f882f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "any((not np.isfinite(value) or value < 0.0 for value in norms))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::7b5386c0f5e68697::1",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(value) or value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::2776f1f61aa53ddd::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::6149324691a26641::2",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::6149324691a26641::2",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::ceb972d0b12db89c::1",
        CandidateClassification.NUMERICAL_GATE,
        "value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::ceb972d0b12db89c::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::raise::ba607a632252eaf9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('warmup X operator norms must be non-empty and non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::f64f954de34e0c17::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(x_operator_norm_margin) or x_operator_norm_margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::d4282ba0b1f481bd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(x_operator_norm_margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::79f7278fa4fdf288::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(x_operator_norm_margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::79f7278fa4fdf288::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(x_operator_norm_margin)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::b6b21b85bcabf315::0",
        CandidateClassification.NUMERICAL_GATE,
        "x_operator_norm_margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::boolean_atom::b6b21b85bcabf315::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x_operator_norm_margin < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::raise::c6398adc99c18149::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('X operator-norm safety margin must be non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::clamp_selector::3c485ea4d4c6e9b4::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(values)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::cd3f603b2fb8ab4d::0",
        CandidateClassification.NUMERICAL_GATE,
        "not certified < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::2d66551cc373e8a1::0",
        CandidateClassification.NUMERICAL_GATE,
        "certified < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::raise::4856d04b75c824f9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'warmup maximum {measured_max} plus margin {margin} and its float64 arithmetic envelope does not certify rho < 1')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::bb30ef6d86bf897d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bases is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::bb30ef6d86bf897d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bases is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::clamp_selector::f64a092de1eaeaf2::0",
        CandidateClassification.STATIC_SELECTOR,
        "max((abs(value) for value in bases))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::6e4233f8716ab021::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "norms is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::6e4233f8716ab021::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "norms is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::clamp_selector::5b3673b7a165b4d3::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(norms)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::decision_predicate::a6a43a5630fe3e1b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "tuple((float(value) for value in measured_rhos))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::decision_predicate::57c724e170630987::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not values or any((not np.isfinite(value) or value < 0.0 for value in values))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::boolean_atom::02429b45018a91c2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not values",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::boolean_atom::362cdb8088bd512d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "any((not np.isfinite(value) or value < 0.0 for value in values))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::decision_predicate::7b5386c0f5e68697::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(value) or value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::boolean_atom::2776f1f61aa53ddd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::predicate_call_atom::6149324691a26641::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::finite_predicate::6149324691a26641::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::compare::ceb972d0b12db89c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::boolean_atom::ceb972d0b12db89c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::raise::1a3db8be99c5bfb7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('retained rho measurements must be non-empty, finite, and non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::compare::f7bd0534d701df1c::0",
        CandidateClassification.NUMERICAL_GATE,
        "value > certificate.certified_rho",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::clamp_selector::3c485ea4d4c6e9b4::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(values)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::compare::d9a05ea226e3d68e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.max_abs_lambda_logdet is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::decision_predicate::d9a05ea226e3d68e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.max_abs_lambda_logdet is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::raise::91975c5561ed7070::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('certificate has no lambda-logdet scale bound')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::decision_predicate::06bac35aae45a6bf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "tuple((abs(float(value)) for value in lambda_logdets))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::decision_predicate::6d359bc09d8b5045::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not values or any((not np.isfinite(value) for value in values))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::boolean_atom::02429b45018a91c2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not values",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::boolean_atom::2b21f0223d591503::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "any((not np.isfinite(value) for value in values))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::predicate_call_atom::6149324691a26641::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::finite_predicate::6149324691a26641::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::raise::485f0f4034778211::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('retained lambda logdets must be non-empty and finite')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::compare::07d13ba7ca0d0c09::0",
        CandidateClassification.NUMERICAL_GATE,
        "value > certificate.max_abs_lambda_logdet",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::clamp_selector::3c485ea4d4c6e9b4::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(values)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::compare::741c99d2b3442c2d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.max_x_operator_norm is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::decision_predicate::741c99d2b3442c2d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.max_x_operator_norm is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::raise::08336c26b9003d91::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('certificate has no |X| operator-norm bound')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::decision_predicate::fc61f410b775692f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "tuple((float(value) for value in absolute_action_norms))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::decision_predicate::57c724e170630987::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not values or any((not np.isfinite(value) or value < 0.0 for value in values))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::boolean_atom::02429b45018a91c2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not values",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::boolean_atom::362cdb8088bd512d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "any((not np.isfinite(value) or value < 0.0 for value in values))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::decision_predicate::7b5386c0f5e68697::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(value) or value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::boolean_atom::2776f1f61aa53ddd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::predicate_call_atom::6149324691a26641::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::finite_predicate::6149324691a26641::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::compare::ceb972d0b12db89c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::boolean_atom::ceb972d0b12db89c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value < 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::raise::4749863cd8c8bd29::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('retained |X| operator norms must be non-empty and non-negative')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::compare::fd7fdbc56498b458::0",
        CandidateClassification.NUMERICAL_GATE,
        "value > certificate.max_x_operator_norm",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::clamp_selector::3c485ea4d4c6e9b4::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(values)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::decision_predicate::3799f389ca3510af::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "tuple(problems)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::decision_predicate::c2c682d2258de459::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not retained",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::raise::8a3f1e659e9fe207::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('retained trace audit needs at least one problem')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::compare::4ae94ae7e3ef3f98::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.trace_order != certificate.order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::decision_predicate::34984b001958dd9c::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.trace_order != certificate.order or _retained_rank_exceeds_certificate(problem, certificate) or problem.exact_power_traces is None or (not _checked_power_traces_match(problem.lambda_matrix, problem.perturbation, problem.exact_power_traces, certificate.order))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::boolean_atom::4ae94ae7e3ef3f98::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.trace_order != certificate.order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::boolean_atom::226e0ff3f3646c5e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_retained_rank_exceeds_certificate(problem, certificate)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::compare::e56c637394ef295f::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.exact_power_traces is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::boolean_atom::e56c637394ef295f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.exact_power_traces is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::boolean_atom::92de97488666195a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not _checked_power_traces_match(problem.lambda_matrix, problem.perturbation, problem.exact_power_traces, certificate.order)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._retained_rank_exceeds_certificate::numerical_premise_call::aff70dff8916fb6c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_algebraic_rank_bound(problem.perturbation, problem.low_rank_factors, problem.lambda_matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._retained_rank_exceeds_certificate::decision_predicate::fc49e3ff31db631b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "True",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._retained_rank_exceeds_certificate::compare::d2be5501ca2b2d8f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rank_bound > certificate.multiplicity",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._retained_rank_exceeds_certificate::decision_predicate::d2be5501ca2b2d8f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "rank_bound > certificate.multiplicity",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_power_traces_match::decision_predicate::77c55f577f741415::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_power_traces_match(lambda_matrix, perturbation, traces, order)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_power_traces_match::numerical_premise_call::77c55f577f741415::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_power_traces_match(lambda_matrix, perturbation, traces, order)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_power_traces_match::decision_predicate::bfdbc42f821a1db3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "False",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::compare::4de160e7956c5d45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::decision_predicate::4de160e7956c5d45::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::raise::9c02c094bcd1faf2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(failure) from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::decision_predicate::b0b3d49680c202d3::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.all(np.isfinite(x)) or not np.isfinite(actual_norm)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::boolean_atom::454612947ebca50f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.all(np.isfinite(x))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::predicate_call_atom::bfc361b4701f87e4::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(x))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::predicate_call_atom::67c48bccbcafebff::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(x)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::finite_predicate::67c48bccbcafebff::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(x)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::boolean_atom::ea3c50128d5224cc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(actual_norm)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::predicate_call_atom::e4236c24a78c1cde::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(actual_norm)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::finite_predicate::e4236c24a78c1cde::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(actual_norm)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::raise::23f09c8490aad598::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(failure)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_lambda_logdet_scale::numerical_premise_call::0c46494d2fa9c2a4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "lambda_logdet(lambda_matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_lambda_logdet_scale::raise::9c02c094bcd1faf2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(failure) from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_lambda_logdet_scale::decision_predicate::ae1c882bb97c3ce8::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(actual_scale)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_lambda_logdet_scale::predicate_call_atom::d14485cd4ec6815a::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(actual_scale)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_lambda_logdet_scale::finite_predicate::d14485cd4ec6815a::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(actual_scale)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_lambda_logdet_scale::raise::23f09c8490aad598::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(failure)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::compare::4ae94ae7e3ef3f98::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.trace_order != certificate.order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::4ae94ae7e3ef3f98::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.trace_order != certificate.order",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::raise::ee3c5c8ad40574a4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime plan order must be the warmup certificate-selected order {certificate.order}; problem carries {problem.trace_order}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::numerical_premise_call::aff70dff8916fb6c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_algebraic_rank_bound(problem.perturbation, problem.low_rank_factors, problem.lambda_matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::compare::ffbfe426624c1b2c::0",
        CandidateClassification.NUMERICAL_GATE,
        "certificate.multiplicity < required_multiplicity",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::ffbfe426624c1b2c::0",
        CandidateClassification.NUMERICAL_GATE,
        "certificate.multiplicity < required_multiplicity",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::raise::4d12f3b64b3475c1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f\"certificate multiplicity {certificate.multiplicity} is below the problem's algebraic rank bound {required_multiplicity}\")",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::compare::d9a05ea226e3d68e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.max_abs_lambda_logdet is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::d9a05ea226e3d68e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.max_abs_lambda_logdet is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::raise::4d7c3c2e461a6292::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime plan requires a warmup max_abs_lambda_logdet scale certificate; pass lambda_logdets to certify_warmup_rho')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::compare::1943e05d2c7883c5::0",
        CandidateClassification.NUMERICAL_GATE,
        "actual_base_scale > certificate.max_abs_lambda_logdet",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::1943e05d2c7883c5::0",
        CandidateClassification.NUMERICAL_GATE,
        "actual_base_scale > certificate.max_abs_lambda_logdet",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::raise::4e91c2a9a3ee0026::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'lambda-logdet scale certificate {certificate.max_abs_lambda_logdet} understates measured absolute logdet {actual_base_scale}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::compare::aeffb9563b9adb5d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.max_x_operator_norm is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::aeffb9563b9adb5d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.max_x_operator_norm is not None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::compare::0cff16fe3f2e1490::0",
        CandidateClassification.NUMERICAL_GATE,
        "actual_norm > certificate.max_x_operator_norm",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::0cff16fe3f2e1490::0",
        CandidateClassification.NUMERICAL_GATE,
        "actual_norm > certificate.max_x_operator_norm",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::raise::1230bc76d497bb5f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'|X| operator-norm certificate {certificate.max_x_operator_norm} understates measured norm {actual_norm}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::b0e44e8a17533440::0",
        CandidateClassification.NUMERICAL_GATE,
        "_validate_strict_rho(problem.lambda_matrix, problem.perturbation, certificate.certified_rho)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._canonical_runtime_probes::raise::0c32221db6af9551::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'captured frozen probes must remain finite after conversion to runtime {runtime_dtype.name}') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._canonical_runtime_probes::decision_predicate::87d9424687bc11bb::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.all(np.isfinite(values))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._canonical_runtime_probes::predicate_call_atom::161234ef6c373d98::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.all(np.isfinite(values))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._canonical_runtime_probes::predicate_call_atom::b992e6871aa5ff32::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(values)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._canonical_runtime_probes::finite_predicate::b992e6871aa5ff32::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(values)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._canonical_runtime_probes::raise::4c7a55fc2856153c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'captured frozen probes must remain finite after conversion to runtime {runtime_dtype.name}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_nonnegative::compare::def44deb716d6bcc::0",
        CandidateClassification.NUMERICAL_GATE,
        "value == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_nonnegative::decision_predicate::21d0c003ac8233e2::0",
        CandidateClassification.NUMERICAL_GATE,
        "value == 0.0 or not np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_nonnegative::boolean_atom::def44deb716d6bcc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_nonnegative::boolean_atom::2776f1f61aa53ddd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_nonnegative::predicate_call_atom::6149324691a26641::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_nonnegative::finite_predicate::6149324691a26641::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_product::compare::e6255f8891ee78cd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_product::decision_predicate::8c0bdd8538a04592::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left == 0.0 or right == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_product::boolean_atom::e6255f8891ee78cd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_product::compare::ed198aef3d282d07::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_product::boolean_atom::ed198aef3d282d07::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_product::compare::884f0c19577beaac::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "product == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_product::decision_predicate::884f0c19577beaac::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "product == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_sum::compare::e6255f8891ee78cd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_sum::decision_predicate::e6255f8891ee78cd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_sum::compare::ed198aef3d282d07::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_sum::decision_predicate::ed198aef3d282d07::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_quotient::compare::8aa06d63faa11937::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "numerator == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_quotient::decision_predicate::8aa06d63faa11937::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "numerator == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_quotient::compare::efb248f5981251c9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "quotient == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_quotient::decision_predicate::efb248f5981251c9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "quotient == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::decision_predicate::08d7acfe400fb258::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "any((value > square_root_maximum for value in magnitudes))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::compare::8e980a3141cedf74::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "value > square_root_maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::raise::ba4515899baa4fce::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise OverflowError",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::decision_predicate::3a95e45f028717bd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(energy) or energy > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::boolean_atom::5fb3b31e4e8796af::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(energy)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::predicate_call_atom::0f934fa1a365f5d6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(energy)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::finite_predicate::0f934fa1a365f5d6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(energy)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::compare::770785106b6509e4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "energy > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::boolean_atom::770785106b6509e4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "energy > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::raise::ba4515899baa4fce::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise OverflowError",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::clamp_selector::4ff5548a53edba13::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(maximum_energy, energy)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::raise::c7388466eed0ba84::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype.name} range requires finite frozen probe energy') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::decision_predicate::d1640f41a06efe07::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(total_energy) or total_energy > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::boolean_atom::4b39a12b5d5f356b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(total_energy)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::predicate_call_atom::ec9578c6de121206::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(total_energy)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::finite_predicate::ec9578c6de121206::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(total_energy)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::compare::0fcd61fc2c50502a::0",
        CandidateClassification.NUMERICAL_GATE,
        "total_energy > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::boolean_atom::0fcd61fc2c50502a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "total_energy > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::raise::51aa6672494f33c9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype.name} range requires finite frozen probe energy')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::compare::e6255f8891ee78cd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::decision_predicate::8c0bdd8538a04592::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left == 0.0 or right == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::boolean_atom::e6255f8891ee78cd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::compare::ed198aef3d282d07::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::boolean_atom::ed198aef3d282d07::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right == 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::decision_predicate::958e52d096caca96::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(left) or not np.isfinite(right) or left > maximum / right",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::boolean_atom::9f6694503fc62bae::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::predicate_call_atom::9d81c8aa824df72e::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::finite_predicate::9d81c8aa824df72e::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::boolean_atom::a094230204f1f810::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(right)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::predicate_call_atom::a24bd6de02c70ffc::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(right)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::finite_predicate::a24bd6de02c70ffc::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(right)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::compare::e3b414eecff3506e::0",
        CandidateClassification.NUMERICAL_GATE,
        "left > maximum / right",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::boolean_atom::e3b414eecff3506e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left > maximum / right",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::raise::41ae40abe4ff21c0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype.name} range cannot certify {quantity}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::decision_predicate::2a13a4e35f93ecfe::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(result) or result > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::boolean_atom::63f0625bbd140dbb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(result)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::predicate_call_atom::3cf98a58825628a4::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(result)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::finite_predicate::3cf98a58825628a4::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(result)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::compare::9b9d8f993db72706::0",
        CandidateClassification.NUMERICAL_GATE,
        "result > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::boolean_atom::9b9d8f993db72706::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "result > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::raise::41ae40abe4ff21c0::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype.name} range cannot certify {quantity}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::decision_predicate::3f0c849b23f9cef1::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(left) or not np.isfinite(right) or right > maximum or (left > maximum - right)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::boolean_atom::9f6694503fc62bae::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::predicate_call_atom::9d81c8aa824df72e::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::finite_predicate::9d81c8aa824df72e::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(left)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::boolean_atom::a094230204f1f810::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(right)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::predicate_call_atom::a24bd6de02c70ffc::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(right)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::finite_predicate::a24bd6de02c70ffc::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(right)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::compare::1b346425452c0d94::0",
        CandidateClassification.NUMERICAL_GATE,
        "right > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::boolean_atom::1b346425452c0d94::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "right > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::compare::aa4ae7fb0f99df59::0",
        CandidateClassification.NUMERICAL_GATE,
        "left > maximum - right",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::boolean_atom::aa4ae7fb0f99df59::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "left > maximum - right",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::raise::41ae40abe4ff21c0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype.name} range cannot certify {quantity}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::decision_predicate::2a13a4e35f93ecfe::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(result) or result > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::boolean_atom::63f0625bbd140dbb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(result)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::predicate_call_atom::3cf98a58825628a4::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(result)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::finite_predicate::3cf98a58825628a4::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(result)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::compare::9b9d8f993db72706::0",
        CandidateClassification.NUMERICAL_GATE,
        "result > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::boolean_atom::9b9d8f993db72706::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "result > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::raise::41ae40abe4ff21c0::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype.name} range cannot certify {quantity}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._gamma_for_count::compare::8da450fabbb947ef::0",
        CandidateClassification.NUMERICAL_GATE,
        "product >= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._gamma_for_count::decision_predicate::8da450fabbb947ef::0",
        CandidateClassification.NUMERICAL_GATE,
        "product >= 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::compare::c10e15779304e86d::0",
        CandidateClassification.NUMERICAL_GATE,
        "x_bound > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::decision_predicate::c10e15779304e86d::0",
        CandidateClassification.NUMERICAL_GATE,
        "x_bound > maximum",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::raise::6f20ab8e34c63ae5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype.name} range cannot certify the X operator-norm bound')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::numerical_premise_call::37668676173a5e52::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_runtime_range_product(image_bound, x_bound, maximum, runtime_dtype, f'the frozen image at power {power}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::numerical_premise_call::4571f8ac420a6891::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_runtime_range_product(image_bound, matvec_factor, maximum, runtime_dtype, f'the rounded frozen image at power {power}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::numerical_premise_call::15cf3e80d21f4833::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_runtime_range_product(total_dot_scale, x_bound, maximum, runtime_dtype, f'the frozen probe-image products at power {power}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::numerical_premise_call::dcdc58ceaaac74f5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_runtime_range_product(total_dot_scale, matvec_factor, maximum, runtime_dtype, f'the rounded frozen probe-image products at power {power}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::numerical_premise_call::6ef3b71993eece08::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_runtime_range_product(total_dot_scale, dot_factor, maximum, runtime_dtype, f'the frozen dot products at power {power}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::numerical_premise_call::23f945e87f4622a1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_runtime_range_product(reduced_sum_bound, reduction_factor, maximum, runtime_dtype, f'the frozen probe reduction at power {power}')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::raise::7e51ba1c179bf3ca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype.name} range cannot certify the frozen correction accumulation') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::numerical_premise_call::6d9734b76eb352db::0",
        CandidateClassification.NUMERICAL_GATE,
        "_runtime_range_product(correction_bound, addition_factor, maximum, runtime_dtype, 'the frozen correction accumulation')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::numerical_premise_call::126f03455393f019::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_two_sum_error(problem.lambda_matrix, problem.perturbation)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::57d1aca8742235de::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime precision validation cannot certify finite Lambda + perturbation arithmetic') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::3204fd05446ce318::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::3204fd05446ce318::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sigma.ndim == 1",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::8fe5d7e22a754fc2::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.any(sigma <= 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::predicate_call_atom::8fe5d7e22a754fc2::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.any(sigma <= 0.0)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::0ca59d408e4f04f6::0",
        CandidateClassification.NUMERICAL_GATE,
        "sigma <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::cac5bb3aafefd926::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime plan requires symmetric positive definite Sigma')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::18513ce0c5e61279::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sign <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::18513ce0c5e61279::0",
        CandidateClassification.NUMERICAL_GATE,
        "sign <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::cac5bb3aafefd926::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime plan requires symmetric positive definite Sigma')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::44ed252067551739::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime precision validation could not compute a finite expected logdet from Lambda + perturbation') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::d8dc3ffb0ede7b65::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(expected)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::predicate_call_atom::99dc85e02dd7a3e8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(expected)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::finite_predicate::99dc85e02dd7a3e8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(expected)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::f2fcbc91cf7e40c2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime precision validation requires a finite expected logdet from Lambda + perturbation')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::9290b32e850d7ae9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime precision validation could not represent a finite expected-logdet ULP') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::ce4340efc14e88e2::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(rounded) or not np.isfinite(ulp)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::boolean_atom::cface858fc9ce6ca::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(rounded)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::predicate_call_atom::72365f29dd057af8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(rounded)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::finite_predicate::72365f29dd057af8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(rounded)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::boolean_atom::e38f9c468a3658ee::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(ulp)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::predicate_call_atom::275beea8cea4f34a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(ulp)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::finite_predicate::275beea8cea4f34a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.isfinite(ulp)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::1c25d0018ea8614b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime precision validation requires a finite expected-logdet ULP')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::4e9bf2538eccc387::0",
        CandidateClassification.NUMERICAL_GATE,
        "ulp > certificate.tolerance",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::4e9bf2538eccc387::0",
        CandidateClassification.NUMERICAL_GATE,
        "ulp > certificate.tolerance",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::ec79aa073352da66::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype} ULP {ulp:.8g} at expected logdet scale {abs(expected):.8g} exceeds certificate tolerance {certificate.tolerance:.8g}; use a wider input/runtime dtype or relax the tolerance')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::2e3fba3ea6b5f4f0::0",
        CandidateClassification.NUMERICAL_GATE,
        "base_scale > maximum_runtime_value",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::2e3fba3ea6b5f4f0::0",
        CandidateClassification.NUMERICAL_GATE,
        "base_scale > maximum_runtime_value",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::425ede1035e2bd67::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype.name} range cannot certify the lambda-logdet scale bound')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::3a459345dea9c31e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::741c99d2b3442c2d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.max_x_operator_norm is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::741c99d2b3442c2d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "certificate.max_x_operator_norm is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::f690d720e7629c11::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('frozen runtime plan requires a warmup max_x_operator_norm certificate for ||abs(X)||_2; pass x_operator_norms to certify_warmup_rho')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::fa6499b78ec05f19::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen_probes is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::fa6499b78ec05f19::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen_probes is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::38aa61897ff10d30::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime precision validation requires canonical frozen probes')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::e123b39c8c4759b2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime precision validation requires a finite frozen series scale') from error",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::17d15162104ee643::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(series_scale)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::predicate_call_atom::2368d001f3e9a883::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(series_scale)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::finite_predicate::2368d001f3e9a883::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(series_scale)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::3b2bc35b925d42ff::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime precision validation requires a finite frozen series scale')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::clamp_selector::c1c2581e9999f113::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(1, (4 * n + 4 + 2 * probe_count) * certificate.order + 4 if frozen else 6 * certificate.order + 4)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::3a459345dea9c31e::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::3a459345dea9c31e::2",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::5d314f0a59d50394::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen_correction_bound is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::5d314f0a59d50394::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen_correction_bound is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::clamp_selector::125add7089c2c0fe::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(series_with_roundoff, frozen_correction_bound)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::3a459345dea9c31e::3",
        CandidateClassification.ORDINARY_VALIDATION,
        "frozen",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::b7c85688cf179fbc::0",
        CandidateClassification.NUMERICAL_GATE,
        "total_error_bound > certificate.tolerance",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::b7c85688cf179fbc::0",
        CandidateClassification.NUMERICAL_GATE,
        "total_error_bound > certificate.tolerance",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::raise::fa9517c7b459a0bd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'runtime {runtime_dtype} analytic tail plus conservative roundoff {total_error_bound:.8g} exceeds certificate tolerance {certificate.tolerance:.8g}; use a wider input/runtime dtype or relax the tolerance')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::compare::e0b263479852baf8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "base.ndim != 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::decision_predicate::e0b263479852baf8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "base.ndim != 0",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::raise::0fffac39ffc6d8a7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime lambda_logdet_value must be a scalar')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::decision_predicate::f3da34068224aef0::0",
        CandidateClassification.NUMERICAL_GATE,
        "expected.itemsize > 4 and (not jax.config.x64_enabled) or any((dtype.kind != 'f' or dtype.itemsize < expected.itemsize for dtype in actual))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::compare::3224ec9d7d999c0f::0",
        CandidateClassification.NUMERICAL_GATE,
        "expected.itemsize > 4",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::decision_predicate::afa73a661affbab4::0",
        CandidateClassification.NUMERICAL_GATE,
        "expected.itemsize > 4 and (not jax.config.x64_enabled)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::boolean_atom::3224ec9d7d999c0f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "expected.itemsize > 4",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::boolean_atom::125ee13393926c47::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not jax.config.x64_enabled",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::boolean_atom::f4ed98b27e34d1d2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "any((dtype.kind != 'f' or dtype.itemsize < expected.itemsize for dtype in actual))",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::compare::4df8c4a204ea9933::0",
        CandidateClassification.NUMERICAL_GATE,
        "dtype.kind != 'f'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::decision_predicate::066e0595e1ad7835::0",
        CandidateClassification.NUMERICAL_GATE,
        "dtype.kind != 'f' or dtype.itemsize < expected.itemsize",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::boolean_atom::4df8c4a204ea9933::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "dtype.kind != 'f'",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::compare::29d03c938376b0b3::0",
        CandidateClassification.NUMERICAL_GATE,
        "dtype.itemsize < expected.itemsize",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::boolean_atom::29d03c938376b0b3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "dtype.itemsize < expected.itemsize",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::raise::508fdb9f4f19345d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f\"runtime values must use real floating precision at least the plan's certified {expected}; got {rendered}. Keep construction and execution inside the same `jax.enable_x64` context\")",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_trace_log_plan::decision_predicate::0c74a65d1edff071::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_validate_plan_certificate(problem, certificate)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_trace_log_plan::compare::e56c637394ef295f::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.exact_power_traces is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_trace_log_plan::decision_predicate::a05753de1a88953b::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.exact_power_traces is None or not _checked_power_traces_match(problem.lambda_matrix, problem.perturbation, problem.exact_power_traces, certificate.order)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_trace_log_plan::boolean_atom::e56c637394ef295f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.exact_power_traces is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_trace_log_plan::boolean_atom::92de97488666195a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not _checked_power_traces_match(problem.lambda_matrix, problem.perturbation, problem.exact_power_traces, certificate.order)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_trace_log_plan::raise::9513295ed3f19966::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('runtime trace plan requires bitwise exact power-trace evidence through the certificate-selected order')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_frozen_trace_log_plan::decision_predicate::0c74a65d1edff071::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "_validate_plan_certificate(problem, certificate)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_frozen_trace_log_plan::compare::655cc937a61c4cc6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "problem.frozen_probes is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_frozen_trace_log_plan::decision_predicate::655cc937a61c4cc6::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.frozen_probes is None",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_frozen_trace_log_plan::raise::9c70a4b59bfc6c75::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('frozen runtime plan requires FrozenProbes')",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_frozen_trace_log_plan::compare::c9ac3a5056f615b8::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.frozen_probes.values.shape[1] != _n(problem.lambda_matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_frozen_trace_log_plan::decision_predicate::c9ac3a5056f615b8::0",
        CandidateClassification.NUMERICAL_GATE,
        "problem.frozen_probes.values.shape[1] != _n(problem.lambda_matrix)",
    ),
    ManifestEntry(
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_frozen_trace_log_plan::raise::7e2e6175869b37b1::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError('frozen runtime probe width must equal the matrix dimension')",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::decision_predicate::2776f1f61aa53ddd::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::predicate_call_atom::6149324691a26641::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::finite_predicate::6149324691a26641::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::decision_predicate::45b422c98efa4d2b::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(floor)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::predicate_call_atom::b410e8e4c610d6de::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(floor)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::finite_predicate::b410e8e4c610d6de::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(floor)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::compare::bb065f25cc0fb571::0",
        CandidateClassification.NUMERICAL_GATE,
        "value <= floor",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::decision_predicate::bb065f25cc0fb571::0",
        CandidateClassification.NUMERICAL_GATE,
        "value <= floor",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::compare::c96b8c0f89a8f758::0",
        CandidateClassification.NUMERICAL_GATE,
        "value >= 1.0 - floor",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::decision_predicate::c96b8c0f89a8f758::0",
        CandidateClassification.NUMERICAL_GATE,
        "value >= 1.0 - floor",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::decision_predicate::15ffb02e90908ffa::0",
        CandidateClassification.NUMERICAL_GATE,
        "not np.isfinite(smallest) or not np.isfinite(largest)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::boolean_atom::632b26896e1cd2d6::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(smallest)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::predicate_call_atom::334dace9ea4ca226::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(smallest)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::finite_predicate::334dace9ea4ca226::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(smallest)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::boolean_atom::8707e3c9768cbc80::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not np.isfinite(largest)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::predicate_call_atom::42b1ecb26daa2662::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(largest)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::finite_predicate::42b1ecb26daa2662::0",
        CandidateClassification.NUMERICAL_GATE,
        "np.isfinite(largest)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::compare::1c27e6520e2bee10::0",
        CandidateClassification.NUMERICAL_GATE,
        "smallest <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::decision_predicate::1c27e6520e2bee10::0",
        CandidateClassification.NUMERICAL_GATE,
        "smallest <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._refuse_graph_single_precision::compare::f9018d03af152846::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "node.name not in graph.latents",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._refuse_graph_single_precision::decision_predicate::f9018d03af152846::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "node.name not in graph.latents",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._refuse_graph_single_precision::decision_predicate::a707fb320611f698::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "isinstance(node, Probabilistic)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>._refuse_graph_single_precision::decision_predicate::7d9bf446fa1cb427::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "jnp.issubdtype(array.dtype, jnp.inexact)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::decision_predicate::adf3fb0023416c76::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "resolve_names(graph, first)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::decision_predicate::8c397938b11458e2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "resolve_names(graph, second)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::decision_predicate::ac0e14e9679b9631::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sorted(set(first_names) & set(second_names))",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::decision_predicate::acd8e469a74e5aa4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "overlap",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::raise::088d28bbf537298b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise GraphError(f'block_coupling needs two disjoint latent blocks, but {overlap} appears in both. Put each latent in exactly one block.')",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::decision_predicate::9c4e5a8de5e29ee8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "check_differentiable(graph, names, values)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::decision_predicate::51743830a84c2b05::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "check_observed_have_locs(graph, values)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::compare::a005e3fe3ce5cc52::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "name not in names",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::linalg_exception_premise::dc1b89c1f16106a5::0",
        CandidateClassification.NUMERICAL_SAFETY,
        "try:\n    l_x = np.linalg.cholesky(f_xx)\n    l_t = np.linalg.cholesky(f_tt)\nexcept np.linalg.LinAlgError as error:\n    raise GraphError('block_coupling needs positive-definite within-block posterior precision at `at`; a Cholesky factor did not exist. Add a proper prior, remove a redundant latent, or choose a finite point.') from error",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::linalg_call_atom::d0ca43b45b317458::0",
        CandidateClassification.NUMERICAL_SAFETY,
        "np.linalg.cholesky(f_xx)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::linalg_call_atom::1e90cfb77ca31791::0",
        CandidateClassification.NUMERICAL_SAFETY,
        "np.linalg.cholesky(f_tt)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::raise::accd403c94fce6b3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise GraphError('block_coupling needs positive-definite within-block posterior precision at `at`; a Cholesky factor did not exist. Add a proper prior, remove a redundant latent, or choose a finite point.') from error",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::58032a79e8a9772d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not graph.latents",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::compare::53aa576700a25e4e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "precision_refusal is not None",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::53aa576700a25e4e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "precision_refusal is not None",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::a1c15f1ba7cd2fc2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "check_differentiable(graph, names, values0)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::compare::91a7d1f968c98657::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x0.dtype != jnp.float64",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::91a7d1f968c98657::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "x0.dtype != jnp.float64",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::compare::49785509827c3a9b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "graph_precision_refusal is not None",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::49785509827c3a9b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "graph_precision_refusal is not None",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::cfd27d21dfcf4f6d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "objective(mode)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::d54c0bad5ddd6460::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "jax.grad(objective)(mode)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::90f1d0f8e005c6e5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "jax.hessian(objective)(mode)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::087e9fc8a8556c2f::0",
        CandidateClassification.NUMERICAL_GATE,
        "bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient)) & jnp.all(jnp.isfinite(hessian)))",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::087e9fc8a8556c2f::0",
        CandidateClassification.NUMERICAL_GATE,
        "bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient)) & jnp.all(jnp.isfinite(hessian)))",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::8f9dae6ee5f5a679::0",
        CandidateClassification.NUMERICAL_GATE,
        "jnp.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::finite_predicate::8f9dae6ee5f5a679::0",
        CandidateClassification.NUMERICAL_GATE,
        "jnp.isfinite(value)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::bitwise_finite_conjunction::5d18a1357f93e06d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient)) & jnp.all(jnp.isfinite(hessian))",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::bitwise_finite_conjunction::66aec1de379f48ab::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient))",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::c100b2a3318e04ae::0",
        CandidateClassification.NUMERICAL_GATE,
        "jnp.all(jnp.isfinite(gradient))",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::c602a9209c1f710a::0",
        CandidateClassification.NUMERICAL_GATE,
        "jnp.isfinite(gradient)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::finite_predicate::c602a9209c1f710a::0",
        CandidateClassification.NUMERICAL_GATE,
        "jnp.isfinite(gradient)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::e2015a81ed24bc08::0",
        CandidateClassification.NUMERICAL_GATE,
        "jnp.all(jnp.isfinite(hessian))",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::5ae94c7be7ed7e6a::0",
        CandidateClassification.NUMERICAL_GATE,
        "jnp.isfinite(hessian)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::finite_predicate::5ae94c7be7ed7e6a::0",
        CandidateClassification.NUMERICAL_GATE,
        "jnp.isfinite(hessian)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::1489f243abd828d2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not finite",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::compare::ed87ffcc51d9fe06::0",
        CandidateClassification.NUMERICAL_GATE,
        "gradient_norm > gradient_floor",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::ed87ffcc51d9fe06::0",
        CandidateClassification.NUMERICAL_GATE,
        "gradient_norm > gradient_floor",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::clamp_selector::64f9a1dc656957b9::0",
        CandidateClassification.STATIC_SELECTOR,
        "max(mode.size, 1)",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::539bc0443a1f70fd::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not smallest > curvature_floor or not largest > absolute_curvature_floor",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::boolean_atom::4146d2d5dd6fb3f9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not smallest > curvature_floor",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::compare::d64931ad93baeec2::0",
        CandidateClassification.NUMERICAL_GATE,
        "smallest > curvature_floor",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::boolean_atom::690223283e153fad::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not largest > absolute_curvature_floor",
    ),
    ManifestEntry(
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::compare::4811937db4eb0e1e::0",
        CandidateClassification.NUMERICAL_GATE,
        "largest > absolute_curvature_floor",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.ReducedGraph._refuse_generic::raise::bbe995f7dce11e31::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "raise GraphError(f'ReducedGraph is NUTS-only: generic compile tried to read {attribute}. Pass this result directly to log_joint, to_numpyro, or nuts; do not pass it to generic compile, whose exact and conditional paths do not read graph-level evidence terms.')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.as_graph::decision_predicate::588e58d3b48cdd4f::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "isinstance(graph, ReducedGraph)",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>._names::decision_predicate::04288db2f8dd3910::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "tuple(values)",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>._names::decision_predicate::a0c3c3a69b9f3dce::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "sorted({name for name in names if names.count(name) > 1})",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>._names::compare::762259ce967b894b::0",
        CandidateClassification.NUMERICAL_GATE,
        "names.count(name) > 1",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>._names::decision_predicate::056ea8effbd5dc22::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "duplicates",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>._names::raise::206d66f80a35f374::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "raise GraphError(f'{argument} repeats {duplicates}. Name each node once; repeated declarations make the reduction boundary ambiguous without changing the graph.')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.check_evidence_nuts_boundary::decision_predicate::6cd3fea15ded835a::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "set(nuts_latents)",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.check_evidence_nuts_boundary::decision_predicate::9a0a94e7ed3e539e::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "[name for name in term.over if name not in nuts]",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.check_evidence_nuts_boundary::compare::702133d443cdba96::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "name not in nuts",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.check_evidence_nuts_boundary::decision_predicate::2c0732d8fb905c99::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "outside",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.check_evidence_nuts_boundary::raise::2a791b72fb9eb2f8::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "raise GraphError(f'evidence_terms[{index}] covers non-NUTS latents {outside}, outside the NUTS block {sorted(nuts)}; its full block is {list(term.over)}. Exact and conditional samplers do not read graph-level density terms, so they would silently omit this likelihood. Add it to nuts_latents (put it in NUTS), or keep the likelihood explicit and do not absorb those observations.')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::414bebc0e844e7b2::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "_names(remove_latents, argument='remove_latents')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::f342106a920e5120::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "_names(absorb_observed, argument='absorb_observed')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::c227f700cb69ec29::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "_names(nuts_latents, argument='nuts_latents')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::89112a622611b773::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "set(graph.latents)",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::273bae94174ebb22::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "[name for name in remove if name not in latent_names]",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::compare::e8e1415a712fee4a::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "name not in latent_names",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::f1e874b62601bd00::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "wrong_remove",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::raise::75823891053587d5::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "raise GraphError(f'remove_latents names {wrong_remove}, which are not latent nodes of this graph; its latents are {list(graph.latents)}. Put only integrated latent names in remove_latents. Observations whose likelihood moved into the term belong in absorb_observed.')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::ebd687dfd17ff9f6::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "set(graph.observed)",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::5a1de76d049600e9::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "[name for name in absorbed if name not in observed_names]",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::compare::b75fdd03b8befab3::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "name not in observed_names",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::2e682b70b64c989c::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "wrong_absorbed",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::raise::364a3fcdd6ebd3aa::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "raise GraphError(f'absorb_observed names {wrong_absorbed}, which are not observed probabilistic nodes of this graph; its observations are {list(graph.observed)}. Name only likelihood nodes here; their deterministic descendants are removed automatically.')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::65481ab457807954::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "set(remove)",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::3f3d981d353a9a83::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "set(absorbed)",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::compare::01449ff00bbcedfd::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "node.name in dropped",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::01449ff00bbcedfd::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "node.name in dropped",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::bc5b8a5b1be2a59e::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "any((parent in dropped for parent in node.parents))",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::compare::9d8d4bfad77c02a3::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "parent in dropped",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::compare::53be361d1464ac2c::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "node.name in unreached_absorbed",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::53be361d1464ac2c::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "node.name in unreached_absorbed",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::77117a8ab2aa6c6b::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "downstream",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::2dbc990b438b7d98::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "not downstream",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::a707fb320611f698::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "isinstance(node, Probabilistic)",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::cd3f1b96f854cba8::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "node.is_latent",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::cd3f1b96f854cba8::1",
        CandidateClassification.STRUCTURAL_CONTROL,
        "node.is_latent",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::raise::aa3f6b5fa9ed5986::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "raise GraphError(f'probabilistic descendant {node.name!r} still depends on the removed region, so its {density} would lose a parent. Add it to {destination} only if evidence_term already contains that density, or do not remove this block.')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::cbdc5cebcc95e3e8::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "unreached_absorbed",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::raise::11e01895cfc0d98b::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "raise GraphError(f'absorb_observed names {unrelated}, which are not descendants of remove_latents or another dropped node. Keep independent likelihoods explicit, or collapse them in a separate reduction whose evidence_term includes those observations.')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::e24226753abfe753::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "tuple((node for node in graph.nodes if node.name not in dropped))",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::compare::746abc25cd32dd23::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "node.name not in dropped",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::080b0b19849eae3e::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "{plate for node in retained_nodes for plate in node.plate}",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::6c8a7f70e68b6f1e::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "Graph(nodes=retained_nodes, plates=tuple((p for p in graph.plates if p.name in retained_plates)), joint_prior=graph.joint_prior, evidence_terms=(*graph.evidence_terms, evidence_term))",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::compare::c929c88db8c9d4d4::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "p.name in retained_plates",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::cb58bf0dd3d3a271::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "set(reduced.latents)",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::8f971023ce87f151::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "[name for name in nuts if name not in retained]",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::compare::da3e0d6051c43ae9::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "name not in retained",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::7cb3b0d1b8febdf1::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "unknown_nuts",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::raise::93f03c26d68dbcd8::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "raise GraphError(f'nuts_latents names {unknown_nuts}, which are not retained latents of the reduced graph; its latents are {list(reduced.latents)}. Pass the NUTS block of the returned graph, after the integrated block has been removed.')",
    ),
    ManifestEntry(
        "src/bayesmith/graph/reduction.py::<module>.reduce_with_evidence::decision_predicate::ef2c804e5d8e7c93::0",
        CandidateClassification.STRUCTURAL_CONTROL,
        "check_evidence_nuts_boundary(reduced, nuts)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.k_cg::decision_predicate::af99b7f76d7e534c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not cg_tol_positive(tol)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.k_cg::raise::b0d76f34b5fd7144::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'the CG tolerance must be strictly positive, got {tol!r}')",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_gap::decision_predicate::db2fc8091e37fa30::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not (math.isfinite(low) and math.isfinite(high))",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_gap::decision_predicate::95f7375a73971930::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(low) and math.isfinite(high)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_gap::boolean_atom::4c70b1f370063e75::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(low)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_gap::predicate_call_atom::4c70b1f370063e75::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(low)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_gap::finite_predicate::4c70b1f370063e75::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(low)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_gap::boolean_atom::2e4bef22275738ae::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(high)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_gap::predicate_call_atom::2e4bef22275738ae::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(high)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_gap::finite_predicate::2e4bef22275738ae::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(high)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_gap::clamp_selector::00a3acd297d48c14::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "max(high, low)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.gap_is_contested::compare::2ef630995dace887::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "gap < CONTESTED_BANDWIDTH",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.gap_is_contested::decision_predicate::2ef630995dace887::0",
        CandidateClassification.NUMERICAL_GATE,
        "gap < CONTESTED_BANDWIDTH",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.timing_noise_in_domain::compare::3ae6110a138fcde9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "tol < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.timing_noise_in_domain::decision_predicate::3ae6110a138fcde9::0",
        CandidateClassification.NUMERICAL_GATE,
        "tol < 1.0",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.cg_tol_positive::compare::135e313228ca2cb8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "tol > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.cg_tol_positive::decision_predicate::135e313228ca2cb8::0",
        CandidateClassification.NUMERICAL_GATE,
        "tol > 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.share_is_dominant::compare::0ce46d21579c5ee0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "share > DOMINANCE_SHARE",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.share_is_dominant::decision_predicate::0ce46d21579c5ee0::0",
        CandidateClassification.NUMERICAL_GATE,
        "share > DOMINANCE_SHARE",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::decision_predicate::1bff0891d01c60f4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not rows",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::decision_predicate::32f6c31c880ee207::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "[row for row in rows if math.isfinite(row.cost_hi)]",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::predicate_call_atom::091466b7ff3d1301::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(row.cost_hi)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::finite_predicate::091466b7ff3d1301::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(row.cost_hi)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::decision_predicate::1489f243abd828d2::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not finite",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::decision_predicate::80f8776a130004c3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "min(finite, key=lambda row: row.cost_hi)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::clamp_selector::80f8776a130004c3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "min(finite, key=lambda row: row.cost_hi)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::compare::016646050977383d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "row is winner",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::decision_predicate::500b5ab4aeb6371c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "row is winner or row.cost_hi == math.inf",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::boolean_atom::016646050977383d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "row is winner",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::compare::e4a31fa2f2f7556f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "row.cost_hi == math.inf",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::boolean_atom::e4a31fa2f2f7556f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "row.cost_hi == math.inf",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::compare::80c93fd713b44c17::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "row.cost_lo < winner.cost_hi",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::decision_predicate::80c93fd713b44c17::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "row.cost_lo < winner.cost_hi",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::decision_predicate::2c2cfedc11a2cd61::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "gap_is_contested(relative_gap(row.cost_hi, winner.cost_hi))",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::predicate_call_atom::2c2cfedc11a2cd61::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "gap_is_contested(relative_gap(row.cost_hi, winner.cost_hi))",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::decision_predicate::29dd74081eba98ad::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "overlap or within_band",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::boolean_atom::acd8e469a74e5aa4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "overlap",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.scoreboard::boolean_atom::d4bba7f806f5434a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "within_band",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>._rows::decision_predicate::b888f6c2e2c8d543::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not timing_noise_in_domain(TIMING_NOISE_TOLERANCE)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>._rows::raise::e9f22179dceec4b4::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'the timing noise tolerance must keep the cost interval positive; got {TIMING_NOISE_TOLERANCE!r}')",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.LadderRecord::decision_predicate::ccafe5e5a6e100af::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "eqx.field(static=True)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.LadderRecord.line::decision_predicate::581829679cad1ee7::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.abstained",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.LadderRecord.line::decision_predicate::e513fbd29886d555::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.contested",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.build_ladder::compare::241115a1fe0ae949::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "strategy == 'declared'",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.build_ladder::decision_predicate::241115a1fe0ae949::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "strategy == 'declared'",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.build_ladder::compare::6b521e0772240877::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.winner is not None",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.build_ladder::decision_predicate::6b521e0772240877::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "verdict.winner is not None",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>._terms::compare::1a86b7af698f765f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "strategy == 'split'",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>._terms::decision_predicate::1a86b7af698f765f::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "strategy == 'split'",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>._terms::compare::85ffac9aeb280b37::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "strategy == 'collapse'",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>._terms::decision_predicate::85ffac9aeb280b37::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "strategy == 'collapse'",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>._terms::compare::cca4ec16a15fde57::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "strategy == 'joint'",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>._terms::decision_predicate::cca4ec16a15fde57::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "strategy == 'joint'",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>._terms::raise::a38e3dafd3b6b67d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f\"there is no cost row called {strategy!r}; the scoreboard prices 'split', 'collapse' and 'joint' and nothing else\")",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.cost_shares::decision_predicate::2992236393fddb89::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not math.isfinite(total) or total <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.cost_shares::boolean_atom::f517e52fd3adde59::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not math.isfinite(total)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.cost_shares::predicate_call_atom::5662962bf2f78843::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(total)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.cost_shares::finite_predicate::5662962bf2f78843::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(total)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.cost_shares::compare::c355389e58b9bab8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "total <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.cost_shares::boolean_atom::c355389e58b9bab8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "total <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.dominant_input::decision_predicate::8785dc9d13a486a9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "share_is_dominant(share)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.dominant_input::predicate_call_atom::8785dc9d13a486a9::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "share_is_dominant(share)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_width::compare::20e56372a0db90b0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "low <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_width::decision_predicate::fcffd963735c9389::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "low <= 0.0 or not math.isfinite(high)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_width::boolean_atom::20e56372a0db90b0::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "low <= 0.0",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_width::boolean_atom::33133f157701773b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "not math.isfinite(high)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_width::predicate_call_atom::2e4bef22275738ae::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(high)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.relative_width::finite_predicate::2e4bef22275738ae::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "math.isfinite(high)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.CostReconciliation.line::decision_predicate::d3a73f193ff70ec8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.within",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.CostReconciliation.line::decision_predicate::18131f330b773013::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.dominant",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.reconcile::decision_predicate::d1a0e30b46aabd23::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "ess",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.reconcile::predicate_call_atom::45125ba6bd0df836::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(row.cost_lo <= measured <= row.cost_hi)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/costs.py::<module>.reconcile::compare::25e7836724f0017d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "row.cost_lo <= measured <= row.cost_hi",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.observed_descendants::decision_predicate::65481ab457807954::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "set(remove)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.observed_descendants::decision_predicate::82d61e99990b0e15::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "frontier",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.observed_descendants::compare::c7b09b0dff6be71a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "name in seen",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.observed_descendants::decision_predicate::c7b09b0dff6be71a::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "name in seen",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.observed_descendants::compare::c7b09b0dff6be71a::1",
        CandidateClassification.ORDINARY_VALIDATION,
        "name in seen",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.pivots_are_finite::decision_predicate::10b0270e03e50a15::0",
        CandidateClassification.NUMERICAL_GATE,
        "jnp.all(jnp.isfinite(pivots))",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.pivots_are_finite::predicate_call_atom::10b0270e03e50a15::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "jnp.all(jnp.isfinite(pivots))",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.pivots_are_finite::predicate_call_atom::6d690e9d76c9cc69::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "jnp.isfinite(pivots)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.pivots_are_finite::finite_predicate::6d690e9d76c9cc69::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "jnp.isfinite(pivots)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.pivots_constrain_block::decision_predicate::a6128093a612fa67::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "jnp.all(pivots[:n_block] > floor)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.pivots_constrain_block::predicate_call_atom::a6128093a612fa67::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "jnp.all(pivots[:n_block] > floor)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/collapse.py::<module>.pivots_constrain_block::compare::fa67000d1d7f01d8::0",
        CandidateClassification.NUMERICAL_GATE,
        "pivots[:n_block] > floor",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.quadratic_cc_crosses_floor::compare::75b08209273ad6f3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "quadratic_cc > floor",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.quadratic_cc_crosses_floor::decision_predicate::75b08209273ad6f3::0",
        CandidateClassification.NUMERICAL_GATE,
        "quadratic_cc > floor",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.ratio_exceeds_declared_multiple::compare::c2fe1a2786d6b1cf::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "ratio > DECLARED_MULTIPLE",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.ratio_exceeds_declared_multiple::decision_predicate::c2fe1a2786d6b1cf::0",
        CandidateClassification.NUMERICAL_GATE,
        "ratio > DECLARED_MULTIPLE",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_is_warranted::decision_predicate::a08dd761fad1ae1b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "switches_away or contested",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_is_warranted::boolean_atom::9048537a0ed27296::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "switches_away",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_is_warranted::boolean_atom::efad76f9e73fe00b::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "contested",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.PilotReport.line::decision_predicate::739739e6af42b17e::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "self.blind_to",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::75b404545378f608::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.asarray(first, dtype=float)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::e1d97a64789fe17c::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "np.asarray(second, dtype=float)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::9c6feeb7981c4b23::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "canonical_correlation(left, right)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::caee75f110316c9d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "augment(left)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::7fd27aa0d9e1d538::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "augment(right)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::5c930dbdbfa396e8::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "canonical_correlation(left_augmented, right_augmented)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::19b6a67428420d97::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "int(left_augmented.shape[1] + right_augmented.shape[1])",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::59c9ecff27fbc7cc::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "sampling_floor(p_aug, n_eff)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::ccc2208f732ab5c3::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "quadratic / linear if linear else math.inf",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::c0c609839761c426::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "linear",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::e7704bea22e851c5::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "quadratic_cc_crosses_floor(quadratic, floor) and ratio_exceeds_declared_multiple(ratio)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::boolean_atom::9ac277108a79b970::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "quadratic_cc_crosses_floor(quadratic, floor)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::boolean_atom::d58e70f86d2fb82d::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "ratio_exceeds_declared_multiple(ratio)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::decision_predicate::8d97d956f399e802::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "vetoed",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.pilot_report::predicate_call_atom::f43ebffc6b4096fb::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "bool(vetoed)",
    ),
    ManifestEntry(
        "src/bayesmith/dispatch/pilot.py::<module>.resolve_switch::decision_predicate::ce2f142c759f7a26::0",
        CandidateClassification.ORDINARY_VALIDATION,
        "report.vetoed",
    ),
    # --- R3 Task 3: src/bayesmith/evaluation/checks.py -------------------
    # Two decision predicates carry registered thresholds (D104 in
    # tail_mass_within_rate, D105 in draws_resolve_the_band); the three
    # structural_control rows are the graph-latent-name check that decides
    # whether the observation mean can be recomputed at all.  Everything
    # else in the module is argument validation.
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.draws_resolve_the_band::compare::1b8786c9fa6a0ea4::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'draws >= DRAW_FLOOR',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.draws_resolve_the_band::decision_predicate::1b8786c9fa6a0ea4::0',
        CandidateClassification.NUMERICAL_GATE,
        'draws >= DRAW_FLOOR',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.tail_mass_within_rate::compare::d2588ff81b262ab1::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'tail_mass >= ALPHA / 2.0',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.tail_mass_within_rate::decision_predicate::d2588ff81b262ab1::0',
        CandidateClassification.NUMERICAL_GATE,
        'tail_mass >= ALPHA / 2.0',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::decision_predicate::28a4863deba7d86e::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "getattr(discrepancy, '__module__', None)",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::decision_predicate::41be2cdc001d5167::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "getattr(discrepancy, '__qualname__', None)",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::decision_predicate::3f6f65dbbf5fdf56::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not isinstance(module_name, str) or not module_name',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::boolean_atom::48ab166ae193fc6a::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not isinstance(module_name, str)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::boolean_atom::6c8bdc711d2afc1a::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not module_name',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::raise::994bb75fdf848c2d::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'the discrepancy {discrepancy!r} has no __module__; a report records where a statistic was defined, not the object itself')",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::decision_predicate::dc9e3460fb99a3dc::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not isinstance(qualname, str) or not qualname',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::boolean_atom::7cf6a67fd63771af::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not isinstance(qualname, str)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::boolean_atom::0e6c08b8e1258815::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not qualname',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::raise::4f451d4dbb6ccf6d::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'the discrepancy {discrepancy!r} has no __qualname__; a report records where a statistic was defined, not the object itself')",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::raise::134a25d113603f1f::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'the discrepancy {identity!r} does not import back: {exc}. A lambda, a function defined inside another function and a REPL definition all have an address and none of them has a home, so recording one would put a name in the artifact that no later reader can turn into the statistic that was actually computed') from exc",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::compare::aee7a6b06c82f706::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'resolved is not discrepancy',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::decision_predicate::aee7a6b06c82f706::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'resolved is not discrepancy',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.discrepancy_identity::raise::e8e054c74ccbe280::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "raise ValueError(f'{identity!r} resolves to {resolved!r}, which is not the discrepancy that was passed ({discrepancy!r}); the identity a report would record names a different statistic')",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._conditioned_units::compare::e4407f710d4859bf::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'mask is None',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._conditioned_units::decision_predicate::e4407f710d4859bf::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'mask is None',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._p_value::compare::0e347f87993748c4::0',
        CandidateClassification.ORDINARY_VALIDATION,
        't_replicated >= t_observed',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._weights::decision_predicate::0370eadc359c11cf::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not isinstance(representation, WeightedDrawsPosterior)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._cell_finding::decision_predicate::1c4d01a90c1de4a5::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'within',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::decision_predicate::b37e5c2377659c21::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'tuple((discrepancy_identity(item) for item in discrepancies))',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::decision_predicate::02d4c37b58ede5ff::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not identities',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::raise::597e2c70c9282154::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "raise TypeError('a predictive check needs at least one discrepancy; with none there is no statistic to compare and the report would be a PASS nobody measured')",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::decision_predicate::895009b0f88d7de5::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not replicated',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::decision_predicate::90de10656f096353::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not draws_resolve_the_band(draws)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::decision_predicate::8dbdf6926d4d0468::0',
        CandidateClassification.STRUCTURAL_CONTROL,
        'tuple((name for name in graph.latents if name not in latents))',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::compare::488aedfb0ffd9bf7::0',
        CandidateClassification.STRUCTURAL_CONTROL,
        'name not in latents',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::decision_predicate::8f3abdef9d2c768f::0',
        CandidateClassification.STRUCTURAL_CONTROL,
        'absent',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::decision_predicate::327187dedc88139f::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not np.all(np.isfinite(t_replicated)) or not np.all(np.isfinite(t_observed))',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::boolean_atom::843c325667fa94c9::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not np.all(np.isfinite(t_replicated))',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::predicate_call_atom::5fa23c7375c450bd::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'np.all(np.isfinite(t_replicated))',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::predicate_call_atom::303e9352877c1658::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'np.isfinite(t_replicated)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::finite_predicate::303e9352877c1658::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'np.isfinite(t_replicated)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::boolean_atom::221b0168be46c694::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not np.all(np.isfinite(t_observed))',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::predicate_call_atom::49756607d525d1ca::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'np.all(np.isfinite(t_observed))',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::predicate_call_atom::aba216bd8616be12::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'np.isfinite(t_observed)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::finite_predicate::aba216bd8616be12::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'np.isfinite(t_observed)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::clamp_selector::b554237a2faec5cb::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'min(p_value, 1.0 - p_value)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::compare::930dba4c7f16b5b9::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "item.code == 'discrepancy_outside_band'",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::compare::5d710c72cee6875d::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'outside == 0',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>._predictive_check::decision_predicate::19ef4f4c3201261a::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'passed',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::decision_predicate::392a55c78b8f7c6a::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not isinstance(graph, Graph)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::raise::c3b30a0ed02e5126::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'raise TypeError(f"posterior_predictive_check\'s graph is a Graph; got {graph!r}")',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::decision_predicate::15d78773004243c6::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not isinstance(predictive, PredictiveResult)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::raise::08bc9d4b1c7dbf3e::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "raise TypeError(f'posterior_predictive_check judges a PredictiveResult; got {type(predictive).__name__}')",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::decision_predicate::4f017fd4bb4a3cb3::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not isinstance(source_posterior, PosteriorResult)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::raise::92ca4f921293748d::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "raise TypeError(f'source_posterior is the PosteriorResult the predictive result names; got {type(source_posterior).__name__}')",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::compare::f658ec4fe3e1d46e::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'source_posterior.meta.artifact_id != reference.artifact_id',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::decision_predicate::daec19e547e75fd2::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'source_posterior.meta.artifact_id != reference.artifact_id or source_posterior.meta.revision != reference.revision',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::boolean_atom::f658ec4fe3e1d46e::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'source_posterior.meta.artifact_id != reference.artifact_id',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::compare::90a2cade023ea052::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'source_posterior.meta.revision != reference.revision',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::boolean_atom::90a2cade023ea052::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'source_posterior.meta.revision != reference.revision',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::raise::6bd3773e4a83cd72::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'raise TypeError("the supplied source posterior is not the version this predictive result\'s source_posterior_ref names; its weights would be some other run\'s")',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::compare::0b073c9f0493b1bd::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'int(array.value.shape[0]) != int(weights.shape[0])',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::decision_predicate::0b073c9f0493b1bd::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'int(array.value.shape[0]) != int(weights.shape[0])',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.posterior_predictive_check::raise::b4d024828eaff27a::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'raise TypeError(f"the predictive result\'s {array.name!r} has {array.value.shape[0]} draws and the source posterior has {weights.shape[0]}; the p-value pairs draw i of one with draw i of the other, so a mismatch is not a rescaling")',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.prior_predictive_check::decision_predicate::392a55c78b8f7c6a::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not isinstance(graph, Graph)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.prior_predictive_check::raise::168fa9c864b0853c::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'raise TypeError(f"prior_predictive_check\'s graph is a Graph; got {graph!r}")',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.prior_predictive_check::decision_predicate::c51431d7497d039d::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'not isinstance(simulation, SimulationResult)',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.prior_predictive_check::raise::0ce28e6800ced8f7::0',
        CandidateClassification.ORDINARY_VALIDATION,
        "raise TypeError(f'prior_predictive_check judges a SimulationResult; got {type(simulation).__name__}')",
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.prior_predictive_check::compare::d11e135d43761c32::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'source is not ParameterSourceKind.PRIOR',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.prior_predictive_check::decision_predicate::d11e135d43761c32::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'source is not ParameterSourceKind.PRIOR',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.prior_predictive_check::decision_predicate::f180bdda3343421b::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'int(simulation.observation_draws[0].value.shape[0]) if simulation.observation_draws else 0',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.prior_predictive_check::decision_predicate::070446107c737df0::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'simulation.observation_draws',
    ),
    ManifestEntry(
        'src/bayesmith/evaluation/checks.py::<module>.prior_predictive_check::decision_predicate::f859e929d7cef145::0',
        CandidateClassification.ORDINARY_VALIDATION,
        'count',
    ),
)

EXPECTED_CANDIDATE_IDS = tuple(entry.candidate_id for entry in EXPECTED_SOURCE_MANIFEST)
