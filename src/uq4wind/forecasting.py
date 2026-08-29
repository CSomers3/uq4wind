"""The feature matrix, the six forecasters, and the expanding-window backtest driving them.

Tabular forecasters share fit(X, y) -> predict(X); those publishing a band also expose
predict_with_interval. ARIMA is univariate and keeps its own fit(y) / forecast_from.
"""

from __future__ import annotations

import itertools
import os
from functools import cache
from typing import NamedTuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

from .conformal import ALPHA, conformal_quantile
from .data import SETTLEMENT_STEP, Alignment, nwp_forecast

HORIZONS = (1, 6, 48)  # settlement periods: 30min, 3h, 24h
ISSUE_EVERY = 2  # forecasts issued hourly on a half-hourly index

MIN_TRAIN_WINDOW = pd.Timedelta("730D")
RETRAIN_EVERY = pd.Timedelta("30D")
MIN_TRAIN_ROWS = 100

ARIMA_ORDER_GRID = list(itertools.product((0, 1, 2), (0, 1), (0, 1, 2)))

#: Searching the grid over the full expanding history at every refit would dominate the backtest.
ARIMA_ORDER_SEARCH_ROWS = 5000

#: Bounding the Kalman re-filter keeps cost per issue time constant.
ARIMA_FILTER_ROWS = 512

POWER_LAG_STEPS = (0, 1, 2, 6, 48)
ROLLING_WINDOW_STEPS = 6

LGBM_PARAMS = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "verbosity": -1}

#: 20 members leave each row out of bag in roughly seven of them.
ENBPI_ENSEMBLE_SIZE = 20

PHYSICAL_RANGE = (0.0, 1.0)

#: The power curve is one-dimensional and monotone, so shrinkage rides on estimator count.
POWER_CURVE_PARAMS = {
    "n_estimators": 300,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "verbosity": -1,
}

#: EMOS start values for (a, b, c, d) in mu = a + b*m, sigma^2 = c^2 + d^2*s^2.
EMOS_INIT = (0.0, 1.0, 0.1, 1.0)
EMOS_MAX_ITER = 200
EMOS_MIN_VARIANCE = 1e-8
EMOS_FOLDS = 5

REFERENCE_AIR_DENSITY = 1.225
GAS_CONSTANT_DRY_AIR = 287.05
CELSIUS_TO_KELVIN = 273.15

#: Pinned rather than the service's "auto" default, which is free to change under a rerun.
TABPFN_MODEL_PATH = "v3_default"
TABPFN_MAX_TRAIN_ROWS = 100_000
TABPFN_N_ESTIMATORS = 8
TABPFN_API_KEY_VAR = "TABPFN_API_KEY"


def build_features(
    power: pd.Series,
    nwp_grid: pd.DataFrame,
    horizon: int,
    step: pd.Timedelta = SETTLEMENT_STEP,
) -> pd.DataFrame:
    """Features for predicting power at `horizon` steps ahead, all knowable at the issue time.

    `power` must be on a gap-free index so positional lags are time-correct.
    """
    features = pd.DataFrame(index=power.index)

    for lag in POWER_LAG_STEPS:
        features[f"power_lag_{lag}"] = power.shift(lag)
    features["power_rolling_mean"] = power.rolling(ROLLING_WINDOW_STEPS).mean()
    features["power_rolling_std"] = power.rolling(ROLLING_WINDOW_STEPS).std()

    features["hour_sin"] = np.sin(2 * np.pi * power.index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * power.index.hour / 24)
    features["doy_sin"] = np.sin(2 * np.pi * power.index.dayofyear / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * power.index.dayofyear / 365.25)

    nwp = nwp_forecast(nwp_grid, power.index, Alignment(lead=horizon * step))
    features["wind_speed_80m"] = nwp["wind_speed_80m"]
    features["wind_speed_80m_cubed"] = nwp["wind_speed_80m"] ** 3

    wind_radians = np.deg2rad(nwp["wind_direction_80m"])
    features["wind_dir_sin"] = np.sin(wind_radians)
    features["wind_dir_cos"] = np.cos(wind_radians)

    features["temperature_80m"] = nwp["temperature_80m"]
    features["pressure_80m"] = nwp["pressure_80m"]
    features["precipitation_surface"] = nwp["precipitation_surface"]

    features["nwp_spread_80m"] = nwp["nwp_spread_80m"]
    features["nwp_spread_range_80m"] = nwp["nwp_spread_p90_80m"] - nwp["nwp_spread_p10_80m"]

    # Spread relative to the level it surrounds: 2 m/s is near-total uncertainty at 3 m/s.
    features["nwp_spread_relative"] = nwp["nwp_spread_80m"] / nwp["wind_speed_80m"].clip(lower=0.1)

    features["nwp_lead_hours"] = nwp["nwp_lead_hours"]
    features["nwp_age_minutes"] = nwp["nwp_age_minutes"]

    return features.dropna()


def build_horizon_targets(power: pd.Series, horizons: tuple[int, ...]) -> pd.DataFrame:
    """One column per horizon: row t holds the value h steps after t."""
    return pd.DataFrame({h: power.shift(-h) for h in horizons})


def _ordered_band(point, lower, upper):
    """Sort a pair of endpoints fitted independently, which cross on a few rows in tens of thousands."""
    return point, np.minimum(lower, upper), np.maximum(lower, upper)


class PersistenceForecaster:
    """The reading at the issue time, bracketed by empirical quantiles of its training residuals."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "PersistenceForecaster":
        residuals = y_train.to_numpy(float) - X_train["power_lag_0"].to_numpy(float)
        self.lower_offset_, self.upper_offset_ = np.quantile(
            residuals, [self.alpha / 2, 1 - self.alpha / 2]
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["power_lag_0"].to_numpy()

    def predict_with_interval(self, X: pd.DataFrame):
        point = self.predict(X)
        return point, point + self.lower_offset_, point + self.upper_offset_


class ARIMAForecaster:
    """Iterative ARIMA with the Gaussian interval implied by its state-space covariance.

    The (p, d, q) order is grid-searched by AIC on the first fit and reused, so the model
    form is selected once from the earliest history.
    """

    def __init__(self, alpha: float = ALPHA, order_grid=ARIMA_ORDER_GRID):
        self.alpha = alpha
        self.order_grid = order_grid
        self.order_ = None

    def _select_order(self, y: np.ndarray):
        best_aic, best_order = np.inf, None
        for order in self.order_grid:
            try:
                aic = ARIMA(y, order=order).fit().aic
            except Exception:
                continue
            if aic < best_aic:
                best_aic, best_order = aic, order
        if best_order is None:
            raise RuntimeError("no ARIMA order in the search grid converged")
        return best_order

    def fit(self, y_train: pd.Series) -> "ARIMAForecaster":
        y = y_train.to_numpy(float)
        if self.order_ is None:
            self.order_ = self._select_order(y[-ARIMA_ORDER_SEARCH_ROWS:])
        self.result_ = ARIMA(y, order=self.order_).fit()
        return self

    def forecast_from(self, history: pd.Series, horizons: tuple[int, ...]) -> dict:
        """{horizon: (point, lower, upper)} past the end of history, parameters held fixed."""
        recent = history.to_numpy(float)[-ARIMA_FILTER_ROWS:]
        forecast = self.result_.apply(recent, refit=False).get_forecast(steps=max(horizons))

        mean = np.asarray(forecast.predicted_mean, dtype=float)
        bounds = np.asarray(forecast.conf_int(alpha=self.alpha), dtype=float)
        return {
            h: (float(mean[h - 1]), float(bounds[h - 1, 0]), float(bounds[h - 1, 1]))
            for h in horizons
        }


class BootstrapEnsembleForecaster:
    """Bootstrapped LightGBM members, banded by a quantile of their out-of-bag residuals.

    Bagging spread alone is the sampling variance of the fitted mean, so it shrinks as the
    members agree whatever the noise level is. The out-of-bag residual is the irreducible
    noise they share. That quantile is fixed for the life of a fit, so this subject
    publishes no flow-dependent band, and the online adaptation it lacks is exactly what
    the wrappers around it supply.
    """

    def __init__(
        self,
        alpha: float,
        ensemble_size: int = ENBPI_ENSEMBLE_SIZE,
        lgbm_params: dict = LGBM_PARAMS,
        seed: int = 0,
    ):
        self.alpha = alpha
        self.ensemble_size = ensemble_size
        self.lgbm_params = lgbm_params
        self.seed = seed

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BootstrapEnsembleForecaster":
        n = len(X_train)
        rng = np.random.default_rng(self.seed)
        targets = y_train.to_numpy(float)

        self.models_ = []
        oob_total, oob_count = np.zeros(n), np.zeros(n)
        for _ in range(self.ensemble_size):
            drawn = rng.integers(0, n, n)
            model = LGBMRegressor(**self.lgbm_params).fit(X_train.iloc[drawn], y_train.iloc[drawn])
            self.models_.append(model)

            held_out = np.setdiff1d(np.arange(n), drawn, assume_unique=False)
            if len(held_out):
                oob_total[held_out] += model.predict(X_train.iloc[held_out])
                oob_count[held_out] += 1

        seen = oob_count > 0
        residuals = targets[seen] - oob_total[seen] / oob_count[seen]
        self.residual_quantile_ = conformal_quantile(np.abs(residuals), self.alpha)
        return self

    def _member_predictions(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray([model.predict(X) for model in self.models_])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._member_predictions(X).mean(axis=0)

    def predict_with_interval(self, X: pd.DataFrame):
        point = self.predict(X)
        return point, point - self.residual_quantile_, point + self.residual_quantile_


class QuantileIntervalForecaster:
    """LightGBM under pinball loss at both tails.

    The point forecast comes from a separate squared-error fit rather than a median tail,
    because point accuracy is reported as RMSE and this target's median sits well under
    its mean.
    """

    def __init__(self, alpha: float, lgbm_params: dict = LGBM_PARAMS):
        self.alpha = alpha
        self.lgbm_params = lgbm_params

    @property
    def tails(self):
        return self.alpha / 2, 1 - self.alpha / 2

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "QuantileIntervalForecaster":
        self.models_ = {
            q: LGBMRegressor(objective="quantile", alpha=q, **self.lgbm_params).fit(
                X_train, y_train
            )
            for q in self.tails
        }
        self.point_ = LGBMRegressor(**self.lgbm_params).fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.point_.predict(X)

    def predict_with_interval(self, X: pd.DataFrame):
        low, high = self.tails
        return _ordered_band(
            self.predict(X), self.models_[low].predict(X), self.models_[high].predict(X)
        )


class EnsemblePowerCurveForecaster:
    """31 GEFS members through a density-corrected, monotone power curve, banded by EMOS.

    The curve is fitted at farm level rather than taken from a datasheet, since the target
    carries wake losses, availability and an 80m-to-hub-height mismatch.

    Raw member quantiles are not the published band: they carry initial-condition
    uncertainty alone and cover roughly half of nominal. The members are post-processed by
    EMOS in the censored form standard for a bounded target, a normal censored to
    PHYSICAL_RANGE whose location is affine in the ensemble mean and whose variance is
    affine in the ensemble variance. The coefficients are fitted out of fold so the band
    being wrapped is not flattered by its own training residuals.
    """

    def __init__(
        self,
        alpha: float,
        member_speeds: pd.DataFrame,
        lgbm_params: dict = POWER_CURVE_PARAMS,
        emos_folds: int = EMOS_FOLDS,
    ):
        self.alpha = alpha
        self.member_speeds = member_speeds
        self.lgbm_params = lgbm_params
        self.emos_folds = emos_folds

    @staticmethod
    def _density(X: pd.DataFrame) -> np.ndarray:
        kelvin = X["temperature_80m"].to_numpy(float) + CELSIUS_TO_KELVIN
        return X["pressure_80m"].to_numpy(float) / (GAS_CONSTANT_DRY_AIR * kelvin)

    def _corrected(self, speed: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        """Speed normalised to reference density; power goes as density times speed cubed."""
        return speed * (self._density(X) / REFERENCE_AIR_DENSITY) ** (1 / 3)

    def _member_matrix(self, X: pd.DataFrame) -> np.ndarray:
        speeds = self.member_speeds.reindex(X.index).to_numpy(float).T
        return self._corrected(speeds, X)

    def _fit_curve(self, speed: np.ndarray, y: pd.Series) -> LGBMRegressor:
        return LGBMRegressor(monotone_constraints=[1], **self.lgbm_params).fit(
            speed.reshape(-1, 1), y
        )

    @staticmethod
    def _through(curve: LGBMRegressor, speeds: np.ndarray) -> np.ndarray:
        return curve.predict(speeds.reshape(-1, 1)).reshape(speeds.shape)

    def _through_curve(self, speeds: np.ndarray) -> np.ndarray:
        return self._through(self.curve_, speeds)

    def _ensemble_moments(self, X: pd.DataFrame):
        """Mean and variance of the members through the curve; the curve is nonlinear, so order matters."""
        power = self._through_curve(self._member_matrix(X))
        return np.nanmean(power, axis=0), np.nanvar(power, axis=0)

    def _out_of_fold_moments(self, X_train: pd.DataFrame, y_train: pd.Series):
        """The same moments, from curves that never saw the row they are scoring.

        Folds are contiguous blocks, so a held-out fold is a stretch of time the curve
        behind it was not trained on.
        """
        speed = self._corrected(X_train["wind_speed_80m"].to_numpy(float), X_train)
        members = self._member_matrix(X_train)
        rows = np.arange(len(X_train))

        mean, variance = np.full(len(X_train), np.nan), np.full(len(X_train), np.nan)
        for fold in np.array_split(rows, self.emos_folds):
            held_in = np.setdiff1d(rows, fold)
            if len(held_in) < MIN_TRAIN_ROWS or len(fold) == 0:
                continue
            curve = self._fit_curve(speed[held_in], y_train.iloc[held_in])
            power = self._through(curve, members[:, fold])
            mean[fold], variance[fold] = np.nanmean(power, axis=0), np.nanvar(power, axis=0)
        return mean, variance

    @staticmethod
    def _censored_nll(theta, m, s2, y) -> float:
        a, b, c, d = theta
        mu = a + b * m
        sigma = np.sqrt(c**2 + d**2 * s2 + EMOS_MIN_VARIANCE)
        low, high = PHYSICAL_RANGE
        # Censoring puts the mass past each bound onto the bound itself, not into a density.
        log_likelihood = np.where(
            y <= low,
            norm.logcdf((low - mu) / sigma),
            np.where(
                y >= high,
                norm.logsf((high - mu) / sigma),
                norm.logpdf((y - mu) / sigma) - np.log(sigma),
            ),
        )
        return -float(log_likelihood.sum())

    def _fit_emos(self, m, s2, y_train: pd.Series):
        """Fit (a, b, c, d) by censored maximum likelihood, or None if the window is too short."""
        y = y_train.to_numpy(float)
        usable = np.isfinite(m) & np.isfinite(s2) & np.isfinite(y)
        if usable.sum() < MIN_TRAIN_ROWS:
            return None
        fitted = minimize(
            self._censored_nll,
            EMOS_INIT,
            args=(m[usable], s2[usable], y[usable]),
            method="L-BFGS-B",
            options={"maxiter": EMOS_MAX_ITER},
        )
        return tuple(fitted.x) if fitted.success else None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "EnsemblePowerCurveForecaster":
        speed = self._corrected(X_train["wind_speed_80m"].to_numpy(float), X_train)
        self.curve_ = self._fit_curve(speed, y_train)
        self.emos_ = self._fit_emos(*self._out_of_fold_moments(X_train, y_train), y_train)
        return self

    def _predictive(self, X: pd.DataFrame):
        m, s2 = self._ensemble_moments(X)
        a, b, c, d = self.emos_
        return a + b * m, np.sqrt(c**2 + d**2 * s2 + EMOS_MIN_VARIANCE)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.emos_ is None:
            return np.nanmean(self._through_curve(self._member_matrix(X)), axis=0)
        return np.clip(self._predictive(X)[0], *PHYSICAL_RANGE)

    def predict_with_interval(self, X: pd.DataFrame):
        if self.emos_ is None:
            power = self._through_curve(self._member_matrix(X))
            lower, upper = np.nanquantile(power, [self.alpha / 2, 1 - self.alpha / 2], axis=0)
            return np.nanmean(power, axis=0), lower, upper

        mu, sigma = self._predictive(X)
        # The censored quantile is the latent quantile clipped, so this band stays in range.
        lower = np.clip(mu + sigma * norm.ppf(self.alpha / 2), *PHYSICAL_RANGE)
        upper = np.clip(mu + sigma * norm.ppf(1 - self.alpha / 2), *PHYSICAL_RANGE)
        return np.clip(mu, *PHYSICAL_RANGE), lower, upper


@cache
def _tabpfn_regressor_class():
    """Authenticate once per process; a backtest would otherwise re-handshake per refit."""
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor

    token = os.environ.get(TABPFN_API_KEY_VAR)
    if token:
        tabpfn_client.set_access_token(token)
    elif not tabpfn_client.get_access_token():
        raise RuntimeError(
            f"TabPFN needs credentials: set {TABPFN_API_KEY_VAR}, or run "
            '`python -c "import tabpfn_client; tabpfn_client.init()"` once to log in.'
        )
    return TabPFNRegressor


class TabPFNForecaster:
    """Hosted TabPFN. Feature rows leave the machine for the service.

    The point forecast is the predictive mean, not the median, for the same reason as
    LightGBM-QR: RMSE asks for the conditional mean.
    """

    def __init__(
        self,
        alpha: float,
        model_path: str = TABPFN_MODEL_PATH,
        max_train_rows: int = TABPFN_MAX_TRAIN_ROWS,
        n_estimators: int = TABPFN_N_ESTIMATORS,
    ):
        self.alpha = alpha
        self.model_path = model_path
        self.max_train_rows = max_train_rows
        self.n_estimators = n_estimators

    @property
    def tails(self):
        return [self.alpha / 2, 1 - self.alpha / 2]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "TabPFNForecaster":
        self.model_ = _tabpfn_regressor_class()(
            model_path=self.model_path, n_estimators=self.n_estimators
        )
        self.model_.fit(
            X_train.iloc[-self.max_train_rows :], y_train.iloc[-self.max_train_rows :]
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(X, output_type="mean"))

    def predict_with_interval(self, X: pd.DataFrame):
        # "main" returns the mean and both tails in one request against a metered service.
        predicted = self.model_.predict(X, output_type="main", quantiles=self.tails)
        lower, upper = np.asarray(predicted["quantiles"])
        return _ordered_band(np.asarray(predicted["mean"]), lower, upper)


class Schedule(NamedTuple):
    """The backtest's timing, which every subject must share to stay comparable."""

    min_train_window: pd.Timedelta = MIN_TRAIN_WINDOW
    retrain_every: pd.Timedelta = RETRAIN_EVERY
    issue_every: int = ISSUE_EVERY


def retrain_windows(index: pd.DatetimeIndex, schedule: Schedule = Schedule()) -> list:
    """(train_end, test_end) pairs for an expanding-window retrain schedule."""
    train_end = index.min() + schedule.min_train_window
    windows = []
    while train_end < index.max():
        test_end = min(train_end + schedule.retrain_every, index.max())
        windows.append((train_end, test_end))
        train_end = test_end
    return windows


def issue_times(
    calendar_index: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    schedule: Schedule = Schedule(),
) -> pd.DatetimeIndex:
    """One test window's issue times, restrided from the full settlement-period calendar.

    Striding a model's own feature index would shift which half-hour lands on the grid and
    desync that model from the others. The lower end is exclusive so consecutive windows
    do not issue their shared boundary twice.
    """
    inside = (calendar_index > train_end) & (calendar_index <= test_end)
    return calendar_index[inside][:: schedule.issue_every]


def run_tabular_backtest(
    features_by_horizon: dict[int, pd.DataFrame],
    targets: pd.DataFrame,
    forecaster_factory,
    calendar_index: pd.DatetimeIndex,
    schedule: Schedule = Schedule(),
    step: pd.Timedelta = SETTLEMENT_STEP,
) -> pd.DataFrame:
    """Expanding-window backtest for a tabular forecaster, refitting on `schedule`.

    Returns issue_time, horizon, y_true, y_pred, plus y_lower/y_upper where published.
    """
    parts = []
    for horizon, features in features_by_horizon.items():
        target = targets[horizon].dropna()
        index = features.index.intersection(target.index).sort_values()

        for train_end, test_end in retrain_windows(calendar_index, schedule):
            # A row at t predicts t + horizon, so it is trainable once that target is observed.
            train_index = index[index <= train_end - horizon * step]
            test_index = issue_times(calendar_index, train_end, test_end, schedule).intersection(
                index
            )
            if len(train_index) < MIN_TRAIN_ROWS or len(test_index) == 0:
                continue

            forecaster = forecaster_factory().fit(
                features.loc[train_index], target.loc[train_index]
            )
            X_test = features.loc[test_index]
            block = {
                "issue_time": test_index,
                "horizon": horizon,
                "y_true": target.loc[test_index].to_numpy(),
            }
            if hasattr(forecaster, "predict_with_interval"):
                point, lower, upper = forecaster.predict_with_interval(X_test)
                block |= {"y_pred": point, "y_lower": lower, "y_upper": upper}
            else:
                block["y_pred"] = forecaster.predict(X_test)
            parts.append(pd.DataFrame(block))

    return pd.concat(parts, ignore_index=True)


def run_arima_backtest(
    power: pd.Series,
    horizons: tuple[int, ...] = HORIZONS,
    alpha: float = ALPHA,
    schedule: Schedule = Schedule(),
) -> pd.DataFrame:
    """Same schedule and output shape as run_tabular_backtest, fitting on the raw series."""
    max_horizon = max(horizons)
    forecaster = ARIMAForecaster(alpha)
    rows = []

    for train_end, test_end in retrain_windows(power.index, schedule):
        history = power.loc[:train_end]
        if history.notna().sum() < MIN_TRAIN_ROWS:
            continue
        forecaster.fit(history)

        window = issue_times(power.index, train_end, test_end, schedule)
        for position in power.index.get_indexer(window):
            if position + max_horizon >= len(power):
                continue
            preds = forecaster.forecast_from(power.iloc[: position + 1], horizons)
            for horizon in horizons:
                y_true = power.iloc[position + horizon]
                if pd.notna(y_true):
                    point, lower, upper = preds[horizon]
                    rows.append(
                        {
                            "issue_time": power.index[position],
                            "horizon": horizon,
                            "y_true": float(y_true),
                            "y_pred": point,
                            "y_lower": lower,
                            "y_upper": upper,
                        }
                    )

    return pd.DataFrame(rows)


def common_issue_times(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Restrict a backtest to the (issue_time, horizon) pairs every model produced, farm by farm.

    Intersecting across farms would demand a pair both farms happened to issue, which is a
    constraint on neither farm's protocol.
    """
    keys = ["issue_time", "horizon"]

    def shared_within(farm_forecasts: pd.DataFrame) -> pd.DataFrame:
        shared = None
        for _, group in farm_forecasts.groupby("model"):
            index = pd.MultiIndex.from_frame(group[keys])
            shared = index if shared is None else shared.intersection(index)
        return farm_forecasts[pd.MultiIndex.from_frame(farm_forecasts[keys]).isin(shared)]

    return pd.concat(
        [shared_within(g) for _, g in forecasts.groupby("farm", sort=False)]
    ).reset_index(drop=True)
