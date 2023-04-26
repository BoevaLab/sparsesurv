import numbers
from collections.abc import Iterable
from functools import partial
from numbers import Real

import numpy as np
from celer import ElasticNetCV as CelerElasticNetCV
from scipy import sparse
from sklearn.linear_model._base import _pre_fit
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import _CVIterableWrapper
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_scalar,
    column_or_1d,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis

from .baseline_hazard_estimation import (
    breslow_estimator_breslow,
    breslow_estimator_efron,
)
from .cv_scoring import basic_cv_fold, linear_cv, vvh_cv_fold
from .gradients import breslow_numba_stable, efron_numba_stable
from .loss import breslow_likelihood_stable, efron_likelihood_stable
from .utils import inverse_transform_survival, transform_survival

GRADIENT_FACTORY = {
    "breslow": breslow_numba_stable,
    "efron": efron_numba_stable,
}

LOSS_FACTORY = {
    "breslow": breslow_likelihood_stable,
    "efron": efron_likelihood_stable,
}


BASELINE_HAZARD_FACTORY = {
    "breslow": breslow_estimator_breslow,
    "efron": breslow_estimator_efron,
}


CVSCORERFACTORY = {
    "linear_predictor": linear_cv,
    "regular": basic_cv_fold,
    "vvh": vvh_cv_fold,
}


COX_REFERENCE = {
    "efron": CoxPHSurvivalAnalysis(
        alpha=0, ties="efron", n_iter=100, tol=1e-09, verbose=0
    ),
    "breslow": CoxPHSurvivalAnalysis(
        alpha=0, ties="breslow", n_iter=100, tol=1e-09, verbose=0
    ),
}


def _path_residuals_preconditioning(
    X,
    y,
    train,
    test,
    path,
    path_params,
    time,
    event,
    alphas=None,
    l1_ratio=1,
    X_order=None,
    dtype=None,
):
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    time_train = time[train]
    time_test = time[test]
    event_train = event[train]
    event_test = event[test]
    if not sparse.issparse(X):
        for array, array_input in (
            (X_train, X),
            (y_train, y),
            (X_test, X),
            (y_test, y),
        ):
            if array.base is not array_input and not array.flags["WRITEABLE"]:
                # fancy indexing should create a writable copy but it doesn't
                # for read-only memmaps (cf. numpy#14132).
                array.setflags(write=True)

    if y.ndim == 1:
        precompute = path_params["precompute"]
    else:
        # No Gram variant of multi-task exists right now.
        # Fall back to default enet_multitask
        precompute = False

    X_train, y_train, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
        X_train,
        y_train,
        None,
        precompute,
        normalize=False,
        fit_intercept=False,
        copy=False,
        sample_weight=None,
    )

    path_params = path_params.copy()
    path_params["Xy"] = Xy
    path_params["X_offset"] = X_offset
    path_params["X_scale"] = X_scale
    path_params["precompute"] = precompute
    path_params["copy_X"] = False
    path_params["alphas"] = alphas
    # needed for sparse cd solver
    path_params["sample_weight"] = None

    if "l1_ratio" in path_params:
        path_params["l1_ratio"] = l1_ratio

    # Do the ordering and type casting here, as if it is done in the path,
    # X is copied and a reference is kept here
    X_train = check_array(
        X_train, accept_sparse="csc", dtype=dtype, order=X_order
    )
    alphas, coefs, _ = path(X_train, y_train, **path_params)
    # del X_train, y_train

    if y.ndim == 1:
        # Doing this so that it becomes coherent with multioutput.
        coefs = coefs[np.newaxis, :, :]
        y_offset = np.atleast_1d(y_offset)
        y_test = y_test[:, np.newaxis]

    train_eta = safe_sparse_dot(X_train, coefs)
    test_eta = safe_sparse_dot(X_test, coefs)
    return (
        train_eta,
        test_eta,
        transform_survival(time=time_train, event=event_train),
        transform_survival(time=time_test, event=event_test),
    )


class ElasticNetCVPreconditioner(CelerElasticNetCV):
    def __init__(
        self,
        model_type,
        l1_ratio=1.0,
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        max_iter=100,
        tol=1e-4,
        cv=None,
        verbose=0,
        max_epochs=50000,
        p0=10,
        prune=True,
        precompute="auto",
        positive=False,
        n_jobs=None,
    ):
        cv = 5 if cv is None else cv
        if isinstance(cv, numbers.Integral):
            cv = StratifiedKFold(cv, shuffle=True, random_state=42)
        elif isinstance(cv, Iterable):
            cv = _CVIterableWrapper(cv)
        elif hasattr(cv, "split"):
            cv = cv
        else:
            raise ValueError(
                "Expected cv to be an integer, sklearn model selection object or an iterable"
            )
        super().__init__(
            l1_ratio=l1_ratio,
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            verbose=verbose,
            max_epochs=max_epochs,
            p0=p0,
            prune=prune,
            precompute=precompute,
            positive=positive,
            n_jobs=n_jobs,
        )
        self.model_type = model_type

    def fit(self, X, y, time, event):
        self._validate_params()

        # This makes sure that there is no duplication in memory.
        # Dealing right with copy_X is important in the following:
        # Multiple functions touch X and subsamples of X and can induce a
        # lot of duplication of memory
        copy_X = self.copy_X and self.fit_intercept

        check_y_params = dict(
            copy=False, dtype=[np.float64, np.float32], ensure_2d=False
        )
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

        n_alphas = len(alphas[0])
        path_params.update({"n_alphas": n_alphas})

        folds = list(self.cv.split(X, (event).astype(int)))
        best_pl_score = -np.inf

        jobs = (
            delayed(_path_residuals_preconditioning)(
                X,
                y,
                train,
                test,
                self.fit_intercept,
                self.path,
                path_params,
                alphas=this_alphas,
                l1_ratio=this_l1_ratio,
                X_order="F",
                dtype=X.dtype.type,
                time=time,
                event=event,
            )
            for this_l1_ratio, this_alphas in zip(l1_ratios, alphas)
            for train, test in folds
        )
        eta_path = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(jobs)

        train_eta_folds, test_eta_folds, train_y_folds, test_y_folds = zip(
            *eta_path
        )
        n_folds = int(len(train_eta_folds) / len(l1_ratios))

        mean_cv_score_l1 = []
        mean_cv_score = []
        sd_cv_score_l1 = []
        sd_cv_score = []

        for i in range(len(l1_ratios)):
            train_eta = train_eta_folds[n_folds * i : n_folds * (i + 1)]
            test_eta = test_eta_folds[n_folds * i : n_folds * (i + 1)]
            train_y = train_y_folds[n_folds * i : n_folds * (i + 1)]
            test_y = test_y_folds[n_folds * i : n_folds * (i + 1)]
            self.cv_score_method = "linear_predictor"
            if self.cv_score_method == "linear_predictor":
                train_eta_method = np.concatenate(train_eta).squeeze()
                test_eta_method = np.concatenate(test_eta).squeeze()
                train_y_method = np.concatenate(train_y).squeeze()
                test_y_method = np.concatenate(test_y).squeeze()
                train_time, train_event = inverse_transform_survival(
                    train_y_method
                )
                test_time, test_event = inverse_transform_survival(
                    test_y_method
                )
                for j in range(len(alphas[i])):
                    likelihood = CVSCORERFACTORY[self.cv_score_method](
                        test_eta_method[:, j],
                        test_time,
                        test_event,
                        LOSS_FACTORY[self.model_type],
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
                        train_eta_method = train_eta[k].squeeze()
                        test_eta_method = test_eta[k].squeeze()
                        train_y_method = train_y[k].squeeze()
                        test_y_method = test_y[k].squeeze()

                        train_time, train_event = inverse_transform_survival(
                            train_y_method
                        )
                        test_time, test_event = inverse_transform_survival(
                            test_y_method
                        )
                        # for j in range(len(alphas[i])):
                        fold_likelihood = CVSCORERFACTORY[
                            self.cv_score_method
                        ](
                            test_eta_method[:, j],
                            test_time,
                            test_event,
                            train_eta_method[:, j],
                            train_time,
                            train_event,
                            LOSS_FACTORY[self.model_type],
                        )
                        if np.isnan(fold_likelihood):
                            test_fold_likelihoods.append(-np.inf)
                        else:
                            test_fold_likelihoods.append(fold_likelihood)
                    mean_cv_score_l1.append(np.mean(test_fold_likelihoods))
                    sd_cv_score_l1.append(np.std(test_fold_likelihoods))
                mean_cv_score.append(mean_cv_score_l1)
                sd_cv_score.append(sd_cv_score_l1)

        self.pl_path_ = mean_cv_score

        for l1_ratio, l1_alphas, pl_alphas, sd_alphas in zip(
            l1_ratios, alphas, mean_cv_score, sd_cv_score
        ):
            i_best_alpha = np.argmax(pl_alphas)

            lambda_1se = pl_alphas[i_best_alpha] - sd_alphas[
                i_best_alpha
            ] / np.sqrt(n_folds)

            i_alpha = np.min(np.where(pl_alphas >= lambda_1se))

            this_best_pl = pl_alphas[i_alpha]
            if this_best_pl > best_pl_score:
                best_alpha = l1_alphas[i_alpha]
                best_l1_ratio = l1_ratio
                best_pl_score = this_best_pl

        self.l1_ratio_ = best_l1_ratio
        self.alpha_ = best_alpha
        if self.alphas is None:
            self.alphas_ = np.asarray(alphas)
            if n_alphas == 1:
                self.alphas_ = self.alphas_[0]

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
        model.fit(X, y)

        if not hasattr(self, "l1_ratio"):
            del self.l1_ratio_

        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

        return self
