import numpy as np
from scipy.sparse import csc_matrix, spdiags
from scipy.linalg import solveh_banded
from joblib import Parallel, delayed
from itertools import product
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.sparse import diags, eye, spdiags


def als_baseline_sparse(y, lam=1e5, p=0.01, n_iter=10, eps=1e-6):
    L = len(y)
    diagonals = [np.ones(L), -2 * np.ones(L), np.ones(L)]
    D = diags(diagonals, [0, -1, -2], shape=(L, L))

    w = np.ones(L)

    for _ in range(n_iter):
        W = diags(w, 0)
        Z = W + lam * (D.T @ D)
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y <= baseline)

    return baseline


def als_baseline_multi(y_array, lam=1e5, p=0.01, n_iter=10, n_jobs=-1):
    z_list = Parallel(n_jobs=n_jobs)(
        delayed(als_baseline_sparse)(y, lam, p, n_iter, 1e-6) for y in y_array
    )
    return np.array(z_list)


def optimize_als_multi(data, lam_range, p_range, n_iter=10, n_jobs=-1, scoring='l2'):
    """
    Optimize ALS baseline correction parameters using cross-validation over multiple time series.

    Parameters:
    - data: 2D numpy array of shape (M, N)
        The raw signals.
    - lam_range: list or numpy array of floats
        Candidate values for the smoothing parameter `lam`.
    - p_range: list or numpy array of floats
        Candidate values for the asymmetry parameter `p`.
    - n_iter: int, default=10
        Number of ALS iterations.
    - n_jobs: int, default=-1
        Number of parallel jobs to run. -1 means using all processors.
    - scoring: str, default='l2'
        Scoring metric to evaluate the residuals ('l2' for L2 norm).

    Returns:
    - best_params: dict
        Dictionary containing the optimal `lam` and `p`.
    - best_baseline: 2D numpy array of shape (M, N)
        The baseline estimated with the optimal parameters.
    """
    M, N = data.shape
    best_score = np.inf
    best_params = None
    best_baseline = None

    param_grid = list(product(lam_range, p_range))

    for lam, p in param_grid:
        print(f"Evaluating performance on (lam, p) = ({lam},{p})")

        baseline = als_baseline_multi(
            data, lam=lam, p=p, n_iter=n_iter, n_jobs=n_jobs)

        residual = data - baseline

        if scoring == 'l2':
            score = np.sum(residual**2)
            print(f"Achieved score of: {score}")
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")

        if score < best_score:
            best_score = score
            best_params = {'lam': lam, 'p': p}
            best_baseline = baseline

    return best_params, best_baseline


def replace_negatives_with_baseline_multi(data, baseline):
    """
    Replace negative values in the original data with baseline-corrected values over multiple time series.

    Parameters:
    - data: 2D numpy array of shape (M, N)
        The raw signals.
    - baseline: 2D numpy array of shape (M, N)
        The estimated baselines.

    Returns:
    - corrected_data: 2D numpy array of shape (M, N)
        The corrected signals where negative values are replaced with the baseline.
    """
    corrected_data = np.where(data < 0, baseline, data)
    return corrected_data


class Preprocessor:
    """Applies the following operations to DF/F data:
        1. Baselining (with optional CV parameter optimization)
        2. Filtering (with optional CV parameter optimization)
        3. Stores processed DF/F as `self.filtered` """

    def __init__(
        self,
        dff,  # DF/F data
        baseline_method='als',  # baselining method (default is ALS)
        # filtering method (default is Savitsky-Golay filter)
        filtering_method='savgol',
        # single parameters --> run once on params; param arrays --> run CV optimization
        baseline_params={},  # baseline method parameters
            filter_params={}):  # filtering method parameters

        self.dff = dff
        self.baselined = dff
        self.filtered = dff

        self.baseline_method = baseline_method
        self.filtering_method = filtering_method

        self.baseline_params = baseline_params
        self.filter_params = filter_params

        self.best_baseline_params = {}
        self.best_filter_params = {}
        self.filter_optimized = False
        self.baseline_optimized = False

        if self.baseline_optimized and len(self.baseline_params.keys()) == 0:
            self.baseline_params = self.best_baseline_params

        if self.filter_optimized and len(self.filter_params.keys()) == 0:
            self.filter_params = self.best_filter_params

    def _correct_baseline(self):
        if self.baseline_method == 'als':
            baseline_arr = als_baseline_multi(
                y_array=self.dff,
                lam=self.baseline_params['lam'],
                p=self.baseline_params['p'],
                n_iter=self.baseline_params['n_iter'],
                n_jobs=self.baseline_params['njobs'])

            baselined_data = replace_negatives_with_baseline_multi(
                data=self.dff, baseline=self.dff - baseline_arr)

            self.baselined = baselined_data
        else:
            raise ValueError(
                "Baseline method not available. You must implement it.")

    def _optimized_correct_baseline(self):
        if self.baseline_method == 'als':
            best_params, best_baseline_arr = optimize_als_multi(
                data=self.dff,
                lam_range=self.baseline_params['lam_range'],
                p_range=self.baseline_params['p_range'],
                n_iter=self.baseline_params['n_iter'],
                n_jobs=self.baseline_params['njobs'],
                scoring=self.baseline_params['scoring'])

            self.best_baseline_params = best_params
            self.best_baseline_params['n_iter'] = self.baseline_params['n_iter']
            self.best_baseline_params['njobs'] = self.baseline_params['njobs']
            self.best_baseline_params['scoring'] = self.baseline_params['scoring']

            self.best_baselines = best_baseline_arr
            self.baseline_optimized = True

            baselined_data = replace_negatives_with_baseline_multi(
                data=self.dff, baseline=self.dff - best_baseline_arr)

            self.baselined = baselined_data

        else:
            raise ValueError(
                "Baseline method not available. You must implement it.")

    def _apply_filter(self):
        if self.filtering_method == 'savgol':
            filtered = savgol_filter(
                x=self.baselined,
                window_length=self.filter_params['window_length'],
                polyorder=self.filter_params['polyorder'],
                deriv=self.filter_params['deriv'])

            self.filtered = filtered

    def _optimized_filter(self):
        pass

    def __call__(self):
        self.apply_baseline = True
        self.apply_filter = True
        self.optimize_baseline = False
        self.optimize_filter = False

        if len(self.baseline_params.keys()) == 0:
            self.apply_baseline = False
        if self.apply_baseline:
            # TODO: fix this so it just checks if we have a list in the values
            print('here')
            if 'p_range' in self.baseline_params:
                print('here')
                self.optimize_baseline = True

        if len(self.filter_params.keys()) == 0:
            self.apply_filter = False
        if self.apply_filter:
            # TODO: fix this so it just checks if we have a list in the values
            if 'window_range' in self.filter_params:
                self.optimize_filter = True

        if self.apply_baseline:
            # baselining
            if self.optimize_baseline:
                self._optimized_correct_baseline()
            else:
                self._correct_baseline()
        if self.apply_filter:
            # filtering
            if self.optimize_filter:
                self._optimized_filter()
            else:
                self._apply_filter()
