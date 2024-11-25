import numpy as np
from scipy.sparse import csc_matrix, spdiags
from scipy.linalg import solveh_banded
from joblib import Parallel, delayed
from itertools import product
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

def als_baseline(y, lam=1e5, p=0.01, niter=10):
    """
    Apply Asymmetric Least Squares baseline correction to a single time series.

    Parameters:
    y : ndarray
        The input time series data (1D array of length N).
    lam : float, optional
        The smoothing parameter (larger values make the baseline smoother).
    p : float, optional
        The asymmetry parameter (between 0 and 1).
    niter : int, optional
        Number of iterations to perform.

    Returns:
    z : ndarray
        The estimated baseline (1D array of length N).
    """
    N = len(y)
    # Precompute DTD diagonals for second-order differences
    DTD_diag0 = np.ones(N) * 6
    DTD_diag0[0] = DTD_diag0[1] = DTD_diag0[-2] = DTD_diag0[-1] = 5
    DTD_diag0[0] = DTD_diag0[-1] = 1

    DTD_diag1 = np.ones(N - 1) * -4
    DTD_diag1[0] = DTD_diag1[-1] = -2

    DTD_diag2 = np.ones(N - 2)

    # Initialize weights
    w = np.ones(N)
    for _ in range(niter):
        # Compute W + λ * D^T D diagonals
        A_diag0 = w + lam * DTD_diag0
        A_diag1 = lam * DTD_diag1
        A_diag2 = lam * DTD_diag2

        # Construct the banded matrix ab for solveh_banded
        ab = np.zeros((3, N))
        ab[0, :] = A_diag0              # Main diagonal
        ab[1, 1:] = A_diag1             # First superdiagonal
        ab[2, 2:] = A_diag2             # Second superdiagonal

        z = solveh_banded(ab, w * y, overwrite_ab=True, overwrite_b=True)

        residuals = y - z
        w = p * (residuals > 0) + (1 - p) * (residuals <= 0)
    return z

def als_baseline_multi(y_array, lam=1e5, p=0.01, niter=10, n_jobs=-1):
    """
    Apply ALS baseline correction to multiple time series in parallel.

    Parameters:
    y_array : ndarray
        The input time series data (2D array of shape (M, N)).
    lam : float, optional
        The smoothing parameter.
    p : float, optional
        The asymmetry parameter.
    niter : int, optional
        Number of iterations to perform.
    n_jobs : int, optional
        The number of jobs to run in parallel (default -1 uses all processors).

    Returns:
    z_array : ndarray
        The estimated baselines (2D array of shape (M, N)).
    """
    # Process each time series in parallel
    z_list = Parallel(n_jobs=n_jobs)(
        delayed(als_baseline)(y, lam, p, niter) for y in y_array
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
        
        baseline = als_baseline_multi(data, lam=lam, p=p, niter=n_iter, n_jobs=n_jobs)
        
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
        dff, # DF/F data
        baseline_method='als', # baselining method (default is ALS)
        filtering_method='savgol', # filtering method (default is Savitsky-Golay filter)
        # single parameters --> run once on params; param arrays --> run CV optimization
        baseline_params={}, # baseline method parameters
        filter_params={}): # filtering method parameters
        
        self.dff = dff
        self.baselined = dff
        self.filtered = dff
        
        self.baseline_method = baseline_method
        self.filtering_method = filtering_method
        
        self.baseline_params = baseline_params
        self.filter_params = filter_params
        
        self.best_baseline_params = {}
        self.best_filter_params = {}
        self.best_baselines = np.array([])
        self.filter_optimized = False
        self.baseline_optimized = False
        
        if self.baseline_optimized and len(self.baseline_params.keys()) == 0:
            self.baseline_params = self.best_baseline_params
            
        if self.filter_optimized and len(self.filter_params.keys()) == 0:
            self.filter_params = self.best_filter_params
    
    def _correct_baseline(self, data):
        if self.baseline_method == 'als':
            baseline_arr = als_baseline_multi(
                y_array=self.dff, 
                lam=self.baseline_params['lam'],
                p=self.baseline_params['p'],
                niter=self.baseline_params['niter'],
                n_jobs=self.baseline_params['njobs'])
            
            baselined_data = replace_negatives_with_baseline_multi(
                data=self.dff, baseline=self.dff - baseline_arr)
            
            self.baselined = baselined_data
        else:
            raise ValueError("Baseline method not available. You must implement it.")
            
            
    def _optimized_correct_baseline(self):
        if self.baseline_method == 'als':
            best_params, best_baseline_arr = optimize_als_multi(                
                y_array=self.dff, 
                lam=self.baseline_params['lam_range'],
                p=self.baseline_params['p_range'],
                niter=self.baseline_params['niter'],
                n_jobs=self.baseline_params['njobs'],
                scoring=self.baseline_params['scoring'])
            
            self.best_baseline_params = best_params
            self.best_baseline_params['niter'] = self.baseline_params['niter']
            self.best_baseline_params['njobs'] = self.baseline_params['njobs']
            self.best_baseline_params['scoring'] = self.baseline_params['scoring']
            
            self.best_baselines = best_baseline_arr
            self.baseline_optimized = True
            
            baselined_data = replace_negatives_with_baseline_multi(
                data=self.dff, baseline=self.dff - best_baseline_arr)
            
            self.baselined = baselined_data
            
        else:
            raise ValueError("Baseline method not available. You must implement it.")
    
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
        apply_baseline = True
        optimize_baseline = False
        optimize_filter = False
        
        if len(self.baseline_params.keys()) == 0:
            apply_baseline = False
        if apply_baseline:
            if isinstance(self.baseline_params[self.baseline_params.keys[0]], (list, np.ndarray)):
                optimize_baseline = True
                
        if isinstance(self.filter_params[self.filter_params.keys[0]], (list, np.ndarray)):
            optimize_filter = True
            
        if apply_baseline:
            # baselining
            if optimize_baseline:
                self._optimized_correct_baseline()
            else:
                self._correct_baseline()
        
        # filtering
        if optimize_filter:
            self._optimized_filter()
        else:
            self._apply_filter()
        