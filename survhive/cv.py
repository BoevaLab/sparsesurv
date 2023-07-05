import numbers
from functools import partial
from numbers import Real
from typing import List, Optional, Union

import celer
import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import effective_n_jobs
from scipy import sparse
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_sample_weight,
    check_consistent_length,
    check_scalar,
    column_or_1d,
)
from typeguard import typechecked

from ._base import SurvivalMixin
from .compat import BASELINE_HAZARD_FACTORY, CVSCORERFACTORY, LOSS_FACTORY
from .utils import _path_predictions, inverse_transform_survival_preconditioning


@typechecked
class PCSurvCV(SurvivalMixin, celer.ElasticNetCV):
    """Parent class to fit preconditioned sparse semiparametric right-censored survival models.

    Parameters
    ----------
    l1_ratio : float or list of float, default=0.5
        Float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For ``l1_ratio = 0``
        the penalty is an L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2
        This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of
        values for l1_ratio is often to put more values close to 1
        (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
        .9, .95, .99, 1]``.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path, used for each l1_ratio.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    cv : int, optional, default=5
        TODO DW

    verbose : bool or int, default=0
        Amount of verbosity.

    max_epochs : int, optional (default=50000)
        Maximum number of coordinate descent epochs when solving a subproblem.

    p0 : int, optional (default=10)
        Number of features in the first working set.

    prune : bool, optional (default=False)
        Whether to use pruning when growing the working sets.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    stratify_cv : bool, optional, default=True
        TODO DW

    seed : int, optional, default=42
        TODO DW

    shuffle_cv : bool, optional, default=False
        TODO DW

    cv_score_method : str, optional, default="linear_predictor"
        TODO DW

    Attributes
    ----------
    TODO DW

    Notes
    -----
    This class is largely adapted from the `ElasticNetCV` implementations
    in `sklearn` and `celer`.

    See Also
    --------
    sklearn.linear_model.ElasticNetCV
    celer.ElasticNetCV
    """

    def __init__(
        self,
        l1_ratio: Union[float, List[float]] = 1.0,
        eps: float = 1e-3,
        n_alphas: int = 100,
        max_iter: int = 100,
        tol: float = 1e-4,
        cv: int = 5,
        verbose: int = 0,
        max_epochs: int = 50000,
        p0: int = 10,
        prune: bool = True,
        n_jobs: Optional[int] = None,
        stratify_cv: bool = True,
        seed: Optional[int] = 42,
        shuffle_cv: bool = False,
        cv_score_method: str = "linear_predictor",
    ):
        super().__init__(
            l1_ratio=l1_ratio,
            eps=eps,
            n_alphas=n_alphas,
            alphas=None,
            fit_intercept=False,
            max_iter=max_iter,
            tol=tol,
            cv=None,
            verbose=verbose,
            max_epochs=max_epochs,
            p0=p0,
            prune=prune,
            positive=False,
            n_jobs=n_jobs,
        )
        self.cv = cv
        self.stratify_cv = stratify_cv
        self.seed = seed
        self.shuffle_cv = shuffle_cv
        self.cv_score_method = cv_score_method

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: Optional[npt.NDArray[np.float64]] = None,
    ):
        """TODO DW"""
        # TODO DW: Add additional parameter checks here and clean
        # up the docs a bit more.
        self._validate_params()
        # This makes sure that there is no duplication in memory.
        # Dealing right with copy_X is important in the following:
        # Multiple functions touch X and subsamples of X and can induce a
        # lot of duplication of memory
        copy_X = self.copy_X and self.fit_intercept

        check_y_params = dict(
            copy=False, dtype=[np.float64, np.float32], ensure_2d=False
        )
        time, event, y = inverse_transform_survival_preconditioning(y)
        sorted_ix = np.argsort(time)
        time = time[sorted_ix]
        event = event[sorted_ix]
        y = y[sorted_ix]
        if isinstance(X, np.ndarray) or sparse.isspmatrix(X):
            # Keep a reference to X
            reference_to_old_X = X
            # Let us not impose fortran ordering so far: it is
            # not useful for the cross-validation loop and will be done
            # by the model fitting itself

            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr. We also want to allow y to be 64 or 32 but check_X_y only
            # allows to convert for 64.
            check_X_params = dict(
                accept_sparse="csc", dtype=[np.float64, np.float32], copy=False
            )
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            if sparse.isspmatrix(X):
                if hasattr(
                    reference_to_old_X, "data"
                ) and not np.may_share_memory(reference_to_old_X.data, X.data):
                    # X is a sparse matrix and has been copied
                    copy_X = False
            elif not np.may_share_memory(reference_to_old_X, X):
                # X has been copied
                copy_X = False
            del reference_to_old_X
        else:
            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr. We also want to allow y to be 64 or 32 but check_X_y only
            # allows to convert for 64.
            check_X_params = dict(
                accept_sparse="csc",
                dtype=[np.float64, np.float32],
                order="F",
                copy=copy_X,
            )
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            copy_X = False

        check_consistent_length(X, y)

        if not self._is_multitask():
            if y.ndim > 1 and y.shape[1] > 1:
                raise ValueError(
                    "For multi-task outputs, use MultiTask%s"
                    % self.__class__.__name__
                )
            y = column_or_1d(y, warn=True)
        else:
            if sparse.isspmatrix(X):
                raise TypeError(
                    "X should be dense but a sparse matrix waspassed"
                )
            elif y.ndim == 1:
                raise ValueError(
                    "For mono-task outputs, use %sCV"
                    % self.__class__.__name__[9:]
                )

        if isinstance(sample_weight, numbers.Number):
            sample_weight = None
        if sample_weight is not None:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=X.dtype
            )

        model = self._get_estimator()

        # All LinearModelCV parameters except 'cv' are acceptable
        path_params = self.get_params()

        # Pop `intercept` that is not parameter of the path function
        path_params.pop("fit_intercept", None)

        if "l1_ratio" in path_params:
            l1_ratios = np.atleast_1d(path_params["l1_ratio"])
            # For the first path, we need to set l1_ratio
            path_params["l1_ratio"] = l1_ratios[0]
        else:
            l1_ratios = [
                1,
            ]
        path_params.pop("cv", None)
        path_params.pop("n_jobs", None)

        alphas = self.alphas
        n_l1_ratio = len(l1_ratios)

        check_scalar_alpha = partial(
            check_scalar,
            target_type=Real,
            min_val=0.0,
            include_boundaries="left",
        )

        if alphas is None:
            alphas = [
                _alpha_grid(
                    X,
                    y,
                    l1_ratio=l1_ratio,
                    fit_intercept=self.fit_intercept,
                    eps=self.eps,
                    n_alphas=self.n_alphas,
                    copy_X=self.copy_X,
                )
                for l1_ratio in l1_ratios
            ]
        else:
            # Making sure alphas entries are scalars.
            for index, alpha in enumerate(alphas):
                check_scalar_alpha(alpha, f"alphas[{index}]")
            # Making sure alphas is properly ordered.
            alphas = np.tile(np.sort(alphas)[::-1], (n_l1_ratio, 1))

        # We want n_alphas to be the number of alphas used for each l1_ratio.
        n_alphas = len(alphas[0])
        path_params.update({"n_alphas": n_alphas})

        path_params["copy_X"] = copy_X
        # We are not computing in parallel, we can modify X
        # inplace in the folds
        if effective_n_jobs(self.n_jobs) > 1:
            path_params["copy_X"] = False

        # init cross-validation generator
        # init cross-validation generator
        cv = check_cv(
            cv=StratifiedKFold(
                n_splits=5, shuffle=self.shuffle_cv, random_state=self.seed
            )
            if self.stratify_cv
            else KFold(
                n_splits=5, shuffle=self.shuffle_cv, random_state=self.seed
            ),
            y=event,
            classifier=self.stratify_cv,
        )
        # Compute path for all folds and compute MSE to get the best alpha
        folds = list(cv.split(X, event))
        best_pl_score = np.inf

        # We do a double for loop folded in one, in order to be able to
        # iterate in parallel on l1_ratio and folds
        jobs = (
            delayed(_path_predictions)(
                X,
                y,
                time,
                event,
                sample_weight,
                train,
                test,
                self.fit_intercept,
                self.path,
                path_params,
                alphas=this_alphas,
                l1_ratio=this_l1_ratio,
                X_order="F",
                dtype=X.dtype.type,
            )
            for this_l1_ratio, this_alphas in zip(l1_ratios, alphas)
            for train, test in folds
        )
        predictions_paths = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(jobs)
        train_eta_folds, test_eta_folds, train_y_folds, test_y_folds = zip(
            *predictions_paths
        )
        n_folds = int(len(test_eta_folds) / len(l1_ratios))

        mean_cv_score_l1 = []
        mean_cv_score = []

        for i in range(len(l1_ratios)):
            train_eta = train_eta_folds[n_folds * i : n_folds * (i + 1)]
            test_eta = test_eta_folds[n_folds * i : n_folds * (i + 1)]
            train_y = train_y_folds[n_folds * i : n_folds * (i + 1)]
            test_y = test_y_folds[n_folds * i : n_folds * (i + 1)]

            if self.cv_score_method == "linear_predictor":
                train_eta_method = np.concatenate(train_eta)
                test_eta_method = np.concatenate(test_eta)
                train_y_method = np.concatenate(train_y)
                test_y_method = np.concatenate(test_y)

                (
                    train_time,
                    train_event,
                    _,
                ) = inverse_transform_survival_preconditioning(train_y_method)
                (
                    test_time,
                    test_event,
                    _,
                ) = inverse_transform_survival_preconditioning(test_y_method)
                for j in range(len(alphas[i])):
                    likelihood = CVSCORERFACTORY[self.cv_score_method](
                        test_linear_predictor=test_eta_method[
                            :, :, j
                        ].squeeze(),
                        test_time=test_time,
                        test_event=test_event,
                        score_function=self.loss,
                    )
                    if np.isnan(likelihood):
                        mean_cv_score_l1.append(-np.inf)
                    else:
                        mean_cv_score_l1.append(likelihood)

                mean_cv_score.append(mean_cv_score_l1)

            else:
                test_fold_likelihoods = []
                for j in range(len(alphas[i])):
                    for k in range(n_folds):
                        train_eta_method = train_eta[k]
                        test_eta_method = test_eta[k]
                        train_y_method = train_y[k].squeeze()
                        test_y_method = test_y[k].squeeze()

                        (
                            train_time,
                            train_event,
                            _,
                        ) = inverse_transform_survival_preconditioning(
                            train_y_method
                        )
                        (
                            test_time,
                            test_event,
                            test_eta_hat,
                        ) = inverse_transform_survival_preconditioning(
                            test_y_method
                        )
                        fold_likelihood = CVSCORERFACTORY[
                            self.cv_score_method
                        ](
                            test_linear_predictor=test_eta_method[
                                :, :, j
                            ].squeeze(),
                            test_time=test_time,
                            test_event=test_event,
                            test_eta_hat=test_eta_hat,
                            train_linear_predictor=train_eta_method[
                                :, :, j
                            ].squeeze(),
                            train_time=train_time,
                            train_event=train_event,
                            score_function=self.loss,
                        )
                        test_fold_likelihoods.append(fold_likelihood)
                    mean_cv_score_l1.append(np.mean(test_fold_likelihoods))
                mean_cv_score.append(mean_cv_score_l1)

        self.pl_path_ = mean_cv_score
        for l1_ratio, l1_alphas, pl_alphas in zip(
            l1_ratios, alphas, mean_cv_score
        ):
            i_best_alpha = np.argmax(mean_cv_score)
            this_best_pl = pl_alphas[i_best_alpha]
            if this_best_pl < best_pl_score:
                best_alpha = l1_alphas[i_best_alpha]
                best_l1_ratio = l1_ratio
                best_pl_score = this_best_pl

        self.l1_ratio_ = best_l1_ratio
        self.alpha_ = best_alpha
        if self.alphas is None:
            self.alphas_ = np.asarray(alphas)
            if n_l1_ratio == 1:
                self.alphas_ = self.alphas_[0]

        # Remove duplicate alphas in case alphas is provided.
        else:
            self.alphas_ = np.asarray(alphas[0])

        # Refit the model with the parameters selected
        common_params = {
            name: value
            for name, value in self.get_params().items()
            if name in model.get_params()
        }
        model.set_params(**common_params)
        model.alpha = best_alpha
        model.l1_ratio = best_l1_ratio
        model.copy_X = copy_X
        precompute = getattr(self, "precompute", None)
        if isinstance(precompute, str) and precompute == "auto":
            model.precompute = False

        if sample_weight is None:
            # MultiTaskElasticNetCV does not (yet) support sample_weight, even
            # not sample_weight=None.
            model.fit(X, y)
        else:
            model.fit(X, y, sample_weight=sample_weight)
        if not hasattr(self, "l1_ratio"):
            del self.l1_ratio_
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.dual_gap_ = model.dual_gap_
        self.n_iter_ = model.n_iter_

        self.train_time_ = time
        self.train_event_ = event
        self.train_eta_ = model.predict(X)
        return self

    def _is_multitask(self):
        """Return whether the model instance in question is a multitask model."""
        return False

    def predict(self, X: npt.NDArray[np.float64]):
        """Calculate linear predictor corresponding to query design matrix X."""
        return X @ self.coef_


@typechecked
class PCPHElasticNetCV(PCSurvCV):
    """TODO DW"""

    def __init__(
        self,
        tie_correction: str = "efron",
        l1_ratio: Union[float, List[float]] = 1.0,
        eps: float = 1e-3,
        n_alphas: int = 100,
        max_iter: int = 100,
        tol: float = 1e-4,
        cv: int = 5,
        verbose: int = 0,
        max_epochs: int = 50000,
        p0: int = 10,
        prune: bool = True,
        n_jobs: Optional[int] = None,
        stratify_cv: bool = True,
        seed: Optional[int] = 42,
        shuffle_cv: bool = False,
        cv_score_method: str = "linear_predictor",
    ):
        super().__init__(
            l1_ratio=l1_ratio,
            eps=eps,
            n_alphas=n_alphas,
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            verbose=verbose,
            max_epochs=max_epochs,
            p0=p0,
            prune=prune,
            n_jobs=n_jobs,
            stratify_cv=stratify_cv,
            seed=seed,
            shuffle_cv=shuffle_cv,
            cv_score_method=cv_score_method,
        )
        if tie_correction not in ["breslow", "efron"]:
            raise ValueError(
                "Expected `tie_corection` to be in ['breslow', 'efron']."
                + f"Found {tie_correction} instead."
            )
        self.loss = LOSS_FACTORY[tie_correction]
        self.cumulative_baseline_hazard = BASELINE_HAZARD_FACTORY[
            tie_correction
        ]

    def predict_cumulative_hazard_function(
        self, X: npt.NDArray[np.float64], time: npt.NDArray[np.float64]
    ) -> pd.DataFrame:
        """TODO DW"""
        if np.min(time) < 0:
            raise ValueError(
                "Times for survival and cumulative hazard prediction must be greater than or equal to zero."
                + f"Minimum time found was {np.min(time)}."
                + "Please remove any times strictly less than zero."
            )
        cumulative_baseline_hazards_times: np.array
        cumulative_baseline_hazards: np.array
        (
            cumulative_baseline_hazards_times,
            cumulative_baseline_hazards,
        ) = self.cumulative_baseline_hazard(
            time=self.train_time_, event=self.train_event_, eta=self.train_eta_
        )
        cumulative_baseline_hazards = np.concatenate(
            [np.array([0.0]), cumulative_baseline_hazards]
        )
        cumulative_baseline_hazards_times: np.array = np.concatenate(
            [np.array([0.0]), cumulative_baseline_hazards_times]
        )
        cumulative_baseline_hazards: np.array = np.tile(
            A=cumulative_baseline_hazards[
                np.digitize(
                    x=time, bins=cumulative_baseline_hazards_times, right=False
                )
                - 1
            ],
            reps=X.shape[0],
        ).reshape((X.shape[0], time.shape[0]))
        log_hazards: np.array = (
            np.tile(
                A=self.predict(X),
                reps=time.shape[0],
            )
            .reshape((time.shape[0], X.shape[0]))
            .T
        )
        cumulative_hazard_function: pd.DataFrame = pd.DataFrame(
            cumulative_baseline_hazards * np.exp(log_hazards),
            columns=time,
        )
        return cumulative_hazard_function


@typechecked
class PCAFTElasticNetCV(PCSurvCV):
    """TODO DW"""

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        l1_ratio: Union[float, List[float]] = 1.0,
        eps: float = 1e-3,
        n_alphas: int = 100,
        max_iter: int = 100,
        tol: float = 1e-4,
        cv: int = 5,
        verbose: int = 0,
        max_epochs: int = 50000,
        p0: int = 10,
        prune: bool = True,
        n_jobs: Optional[int] = None,
        stratify_cv: bool = True,
        seed: Optional[int] = 42,
        shuffle_cv: bool = False,
        cv_score_method: str = "linear_predictor",
    ):
        super().__init__(
            l1_ratio=l1_ratio,
            eps=eps,
            n_alphas=n_alphas,
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            verbose=verbose,
            max_epochs=max_epochs,
            p0=p0,
            prune=prune,
            n_jobs=n_jobs,
            stratify_cv=stratify_cv,
            seed=seed,
            shuffle_cv=shuffle_cv,
            cv_score_method=cv_score_method,
        )
        self.bandwidth = bandwidth
        self.loss = LOSS_FACTORY["aft"]
        self.cumulative_baseline_hazard = BASELINE_HAZARD_FACTORY["aft"]

    def predict_cumulative_hazard_function(
        self, X: npt.NDArray[np.float64], time: npt.NDArray[np.float64]
    ) -> pd.DataFrame:
        """TODO DW"""
        if np.min(time) < 0:
            raise ValueError(
                "Times for survival and cumulative hazard prediction must be greater than or equal to zero."
                + f"Minimum time found was {np.min(time)}."
                + "Please remove any times strictly less than zero."
            )
        return self.cumulative_baseline_hazard(
            time_query=time,
            eta_query=self.predict(X=X),
            time_train=self.train_time_,
            event_train=self.train_event_,
            eta_train=self.train_eta_,
        )


@typechecked
class PCEHElasticNetCV(PCSurvCV):
    """TODO DW"""

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        l1_ratio: Union[float, List[float]] = 1.0,
        eps: float = 1e-3,
        n_alphas: int = 100,
        max_iter: int = 100,
        tol: float = 1e-4,
        cv: int = 5,
        verbose: int = 0,
        max_epochs: int = 50000,
        p0: int = 10,
        prune: bool = True,
        n_jobs: Optional[int] = None,
        stratify_cv: bool = True,
        seed: Optional[int] = 42,
        shuffle_cv: bool = False,
        cv_score_method: str = "linear_predictor",
    ):
        super().__init__(
            l1_ratio=l1_ratio,
            eps=eps,
            n_alphas=n_alphas,
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            verbose=verbose,
            max_epochs=max_epochs,
            p0=p0,
            prune=prune,
            n_jobs=n_jobs,
            stratify_cv=stratify_cv,
            seed=seed,
            shuffle_cv=shuffle_cv,
            cv_score_method=cv_score_method,
        )
        self.bandwidth = bandwidth
        self.loss = LOSS_FACTORY["eh"]
        self.cumulative_baseline_hazard = BASELINE_HAZARD_FACTORY["eh"]

    def path(self, X, y, alphas, coef_init=None, **kwargs):
        """Compute Lasso path with Celer.

        Function taken as-is from celer for compatibility with parent class.

        See Also
        --------
        celer.homotopy.mtl_path
        celer.dropin_sklearn.MultiTaskLassoCV
        """

        alphas, coefs, dual_gaps = celer.homotopy.mtl_path(
            X,
            y,
            alphas=alphas,
            coef_init=coef_init,
            max_iter=self.max_iter,
            max_epochs=self.max_epochs,
            p0=self.p0,
            verbose=self.verbose,
            tol=self.tol,
            prune=self.prune,
        )

        return alphas, coefs, dual_gaps

    def predict(self, X: npt.NDArray[np.float64]):
        """Calculate linear predictor corresponding to query design matrix X."""
        return X @ self.coef_.T

    def _is_multitask(self):
        """Return whether the model instance in question is a multitask model."""
        return True

    def predict_cumulative_hazard_function(
        self, X: npt.NDArray[np.float64], time: npt.NDarray[np.float64]
    ) -> pd.DataFrame:
        """TODO DW"""
        if np.min(time) < 0:
            raise ValueError(
                "Times for survival and cumulative hazard prediction must be greater than or equal to zero."
                + f"Minimum time found was {np.min(time)}."
                + "Please remove any times strictly less than zero."
            )
        return self.cumulative_baseline_hazard(
            time_query=time,
            eta_query=self.predict(X=X),
            time_train=self.train_time_,
            event_train=self.train_event_,
            eta_train=self.train_eta_,
        )
