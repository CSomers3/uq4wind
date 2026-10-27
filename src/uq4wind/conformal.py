"""Conformal rules, and the loop that runs them over one forecast stream.

A rule takes a base interval and widens it by a quantile of recent conformity scores.
It is told the outcome later, once the observation was actually visible.
"""

from __future__ import annotations

from collections import defaultdict, deque
from functools import partial
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

ALPHA = 0.1

CALIB_WINDOW = 1000
WARMUP_FRAC = 0.1

ACI_GAMMAS = (0.01, 0.05)
AGACI_DTACI_GAMMAS = np.linspace(0.001, 0.05, 50)
DTACI_WINDOW = 50
NEXCP_RHO = 0.99

LEVEL_EPS = 1e-10

#: A wind band thinner than this uses the pooled buffer instead of its own.
MONDRIAN_MIN_SCORES = 100


class Interval(NamedTuple):
    lower: float
    upper: float

    def widened(self, q: float) -> "Interval":
        if q <= (self.lower - self.upper) / 2:
            midpoint = (self.lower + self.upper) / 2
            return Interval(midpoint, midpoint)
        return Interval(self.lower - q, self.upper + q)


def conformal_quantile(scores: NDArray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th score, with rank capped to [1, n]."""
    n = len(scores)
    rank = int(np.clip(np.ceil((n + 1) * (1 - alpha)), 1, n))
    return float(np.partition(scores, rank - 1)[rank - 1])


def weighted_conformal_quantile(scores: NDArray, alpha: float, rho: float) -> float:
    """Conformal quantile under geometric weights rho**age, oldest lightest."""
    n = len(scores)
    weights = rho ** np.arange(n - 1, -1, -1)

    order = np.argsort(scores)
    ordered_scores, ordered_weights = scores[order], weights[order]
    # The test point carries unit weight, so a light calibration set cannot reach 1 - alpha.
    cumulative = np.cumsum(ordered_weights) / (weights.sum() + 1.0)

    position = int(np.searchsorted(cumulative, 1 - alpha))
    return float(ordered_scores[min(position, n - 1)])


def _clip_level(level):
    return np.clip(level, LEVEL_EPS, 1 - LEVEL_EPS)


def _step_level(level, gamma, alpha, miss):
    """Tighten after a miss, loosen after a hit. Unclipped; only the quantile lookup clips."""
    return level + gamma * (alpha - miss)


def _miscoverage_level(scores: NDArray, score: float) -> float:
    """The level at which the interval would exactly touch this observation."""
    return float(np.count_nonzero(scores > score) / len(scores))


def _pinball(residual, level):
    return residual * (level - (residual < 0))


class _Fixed:
    """Base for rules whose target level never moves."""

    stratified = False

    def update(self, score: float, y: float) -> None:
        pass


class _Adaptive:
    """Base for rules that learn from feedback, graded on the interval they issued."""

    stratified = False

    def __init__(self):
        self._issued: deque = deque()


class OSSCP(_Fixed):
    """Online split conformal at a fixed level."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def interval(self, scores: NDArray, base: Interval) -> Interval:
        return base.widened(conformal_quantile(scores, self.alpha))


class MondrianOSSCP(OSSCP):
    """OSSCP calibrated within the row's wind band; the loop supplies that band's scores."""

    stratified = True


class NexCP(_Fixed):
    """Fixed level over geometrically reweighted scores."""

    def __init__(self, alpha: float, rho: float = NEXCP_RHO):
        self.alpha = alpha
        self.rho = rho

    def interval(self, scores: NDArray, base: Interval) -> Interval:
        return base.widened(weighted_conformal_quantile(scores, self.alpha, self.rho))


class ACI(_Adaptive):
    """Adaptive conformal inference at learning rate gamma."""

    def __init__(self, alpha: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha

    def interval(self, scores: NDArray, base: Interval) -> Interval:
        q = conformal_quantile(scores, float(_clip_level(self.alpha_t)))
        q = max(q, (base.lower - base.upper) / 2)
        self._issued.append(q)
        return base.widened(q)

    def update(self, score: float, y: float) -> None:
        miss = float(score > self._issued.popleft())
        self.alpha_t = float(_step_level(self.alpha_t, self.gamma, self.alpha, miss))


class _BOA:
    """Bernstein Online Aggregation over k experts."""

    def __init__(self, k: int, eps: float = 1e-3):
        self.k = k
        self.eps = eps
        self.probs = np.full(k, 1.0 / k)
        self.regret = np.zeros(k)
        self.sq_regret = np.zeros(k)
        self.max_regret = np.zeros(k)
        self.etas = np.zeros(k)

    def predict(self, expert_values: NDArray) -> float:
        return float(self.probs @ expert_values)

    def update(self, expert_losses: NDArray, aggregate_loss: float) -> None:
        instantaneous = expert_losses - aggregate_loss
        self.sq_regret += instantaneous**2
        self.max_regret = np.maximum(self.max_regret, np.abs(instantaneous))
        ranges = 2 ** (np.ceil(np.log2(self.max_regret + self.eps)) + 1)

        self.regret += 0.5 * (
            instantaneous * (1 + self.etas * instantaneous)
            + ranges * (self.etas * instantaneous > 0.5)
        )

        with np.errstate(divide="ignore"):
            self.etas = np.minimum(1 / ranges, np.sqrt(np.log(self.k) / self.sq_regret))

        scaled = self.etas * self.regret
        weights = self.etas * np.exp(-scaled + np.min(scaled))
        self.probs = weights / weights.sum()


class AgACI(_Adaptive):
    """One ACI expert per learning rate, endpoints aggregated by BOA under pinball loss."""

    def __init__(self, alpha: float, gammas: NDArray, eps: float = 1e-3):
        super().__init__()
        self.alpha = alpha
        self.gammas = np.asarray(gammas, dtype=float)
        self.expert_alphas = np.full(len(self.gammas), alpha)

        self.lower_boa = _BOA(len(self.gammas), eps)
        self.upper_boa = _BOA(len(self.gammas), eps)

    def interval(self, scores: NDArray, base: Interval) -> Interval:
        quantiles = np.array(
            [conformal_quantile(scores, a) for a in _clip_level(self.expert_alphas)]
        )
        quantiles = np.maximum(quantiles, (base.lower - base.upper) / 2)
        lowers, uppers = base.lower - quantiles, base.upper + quantiles

        issued = Interval(self.lower_boa.predict(lowers), self.upper_boa.predict(uppers))
        self._issued.append((quantiles, lowers, uppers, issued))
        return issued

    def update(self, score: float, y: float) -> None:
        quantiles, lowers, uppers, issued = self._issued.popleft()

        self.lower_boa.update(
            _pinball(y - lowers, self.alpha / 2), _pinball(y - issued.lower, self.alpha / 2)
        )
        self.upper_boa.update(
            _pinball(y - uppers, 1 - self.alpha / 2), _pinball(y - issued.upper, 1 - self.alpha / 2)
        )

        misses = (score > quantiles).astype(float)
        self.expert_alphas = _step_level(self.expert_alphas, self.gammas, self.alpha, misses)


class DtACI(_Adaptive):
    """One expert level per step, chosen by exponential reweighting over a window."""

    def __init__(self, alpha: float, gammas: NDArray, window: int = DTACI_WINDOW):
        super().__init__()
        self.alpha = alpha
        self.gammas = np.asarray(gammas, dtype=float)

        k = len(self.gammas)
        self.weights = np.ones(k)
        self.expert_alphas = np.full(k, alpha)

        denom = (1 - alpha) ** 2 * alpha**3 + alpha**2 * (1 - alpha) ** 3
        self.eta = np.sqrt(3 / window) * np.sqrt((np.log(k * window) + 2) / denom)
        self.sigma = 1 / (2 * window)

    def interval(self, scores: NDArray, base: Interval) -> Interval:
        probs = self.weights / self.weights.sum()
        alpha_t = float(_clip_level(probs @ self.expert_alphas))

        quantiles = np.array([
            max(conformal_quantile(scores, a), (base.lower - base.upper) / 2)
            for a in _clip_level(self.expert_alphas)
        ])
        self._issued.append((scores, self.expert_alphas.copy(), quantiles))
        return base.widened(conformal_quantile(scores, alpha_t))

    def update(self, score: float, y: float) -> None:
        scores, issued_alphas, quantiles = self._issued.popleft()
        beta = _miscoverage_level(scores, score)

        losses = self.alpha * (beta - issued_alphas) - np.minimum(0, beta - issued_alphas)
        pooled = self.weights * np.exp(-self.eta * losses)
        self.weights = (1 - self.sigma) * pooled / pooled.sum() + self.sigma / len(self.gammas)

        misses = (score > quantiles).astype(float)
        self.expert_alphas = _step_level(self.expert_alphas, self.gammas, self.alpha, misses)


def default_methods(alpha: float = ALPHA) -> dict:
    """The seven wrappers scored in the paper."""
    methods = {"osscp": partial(OSSCP, alpha), "nexcp": partial(NexCP, alpha, NEXCP_RHO)}
    for gamma in ACI_GAMMAS:
        methods[f"aci_{gamma}"] = partial(ACI, alpha, gamma)
    methods["agaci"] = partial(AgACI, alpha, AGACI_DTACI_GAMMAS)
    methods["dtaci"] = partial(DtACI, alpha, AGACI_DTACI_GAMMAS, DTACI_WINDOW)
    methods["mondrian"] = partial(MondrianOSSCP, alpha)
    return methods


def run_online_conformal(
    y: NDArray,
    y_pred: NDArray,
    methods: dict,
    lower_base: NDArray | None = None,
    upper_base: NDArray | None = None,
    warmup_frac: float = WARMUP_FRAC,
    calib_window: int = CALIB_WINDOW,
    feedback_delay: int = 1,
    strata: NDArray | None = None,
    issue_steps: NDArray | None = None,
) -> dict[str, tuple[NDArray, NDArray]]:
    """Run every rule over one stream, each widening the base interval by a quantile of

        score_t = max(lower_base_t - y_t, y_t - upper_base_t)

    Equal bases give split conformal on the absolute residual; two tails give CQR.
    `feedback_delay` holds each score back until its error was observable. `issue_steps`
    gives elapsed issue-grid steps when rows are missing; otherwise rows are consecutive.

    Returns {label: (lower, upper)}, each covering y[int(len(y) * warmup_frac):].
    """
    y, y_pred = np.asarray(y, dtype=float), np.asarray(y_pred, dtype=float)
    if (lower_base is None) != (upper_base is None):
        raise ValueError("lower_base and upper_base must be given together")
    lower_base = y_pred if lower_base is None else np.asarray(lower_base, dtype=float)
    upper_base = y_pred if upper_base is None else np.asarray(upper_base, dtype=float)

    warmup = int(len(y) * warmup_frac)
    if not 1 <= warmup < len(y):
        raise ValueError("warmup must leave both calibration and evaluation rows")
    if feedback_delay < 1 or calib_window < 1:
        raise ValueError("feedback_delay and calib_window must be positive")
    steps = np.arange(len(y)) if issue_steps is None else np.asarray(issue_steps)
    if steps.shape != y.shape or not np.isfinite(steps).all() or np.any(np.diff(steps) <= 0):
        raise ValueError("issue_steps must be finite and strictly increasing, one per outcome")
    release = steps + feedback_delay
    seed_end = int(np.searchsorted(release[:warmup], steps[warmup], side="right"))
    if seed_end == 0:
        raise ValueError("no warmup outcomes are observable at the first scored issue")

    stream_scores = np.maximum(lower_base - y, y - upper_base)
    missing = int((~np.isfinite(stream_scores)).sum())
    if missing:
        raise ValueError(f"{missing} of {len(y)} conformity scores are not finite")

    arms = {}
    for label, factory in methods.items():
        rule = factory()
        if rule.stratified and strata is None:
            raise ValueError(f"rule {label!r} calibrates within strata, but none were given")

        buffer = np.empty(len(y), dtype=float)
        buffer[:seed_end] = stream_scores[:seed_end]
        n_scores = seed_end

        by_stratum: dict[object, deque] = defaultdict(lambda: deque(maxlen=calib_window))
        if rule.stratified:
            for i in range(seed_end):
                by_stratum[strata[i]].append(float(stream_scores[i]))

        pending = deque(
            (release[i], i, float(stream_scores[i]), float(y[i]))
            for i in range(seed_end, warmup)
        )
        lowers, uppers = [], []

        for t in range(warmup, len(y)):
            while pending and pending[0][0] <= steps[t]:
                _, i, score, y_obs = pending.popleft()
                if i >= warmup:
                    rule.update(score, y_obs)
                buffer[n_scores] = score
                n_scores += 1
                if rule.stratified:
                    by_stratum[strata[i]].append(score)

            calibration = buffer[max(0, n_scores - calib_window) : n_scores]
            if rule.stratified:
                local = by_stratum[strata[t]]
                if len(local) >= MONDRIAN_MIN_SCORES:
                    calibration = np.fromiter(local, dtype=float, count=len(local))

            lower, upper = rule.interval(calibration, Interval(lower_base[t], upper_base[t]))
            lowers.append(lower)
            uppers.append(upper)

            pending.append((release[t], t, float(stream_scores[t]), float(y[t])))

        arms[label] = (np.array(lowers), np.array(uppers))
    return arms
