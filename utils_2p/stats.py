import numpy as np
from scipy.stats import t


def norm01(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-5)


def get_norm01_params(data):
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    scale = 1 / (data_max - data_min)
    offset = -data_min / (data_max - data_min)
    return scale, offset, data_min, data_max


def get_mean_sem(
    data,
    method_m="mean",
    method_s="standard error",
    zero_start=False,
):
    reshaped = data.reshape(-1, data.shape[-1])
    if method_m == "mean":
        mean = np.nanmean(reshaped, axis=0)
    elif method_m == "median":
        mean = np.nanmedian(reshaped, axis=0)
    else:
        raise ValueError(f"Unsupported method_m: {method_m}")

    mean = mean - mean[0] if zero_start else mean
    std = np.nanstd(reshaped, axis=0)
    count = np.nansum(~np.isnan(reshaped), axis=0)

    if method_s == "confidence interval":
        spread = t.ppf(0.975, count - 1) * std / np.sqrt(count)
    elif method_s == "prediction interval":
        spread = t.ppf(0.975, count - 1) * std * np.sqrt(1 + 1 / count)
    elif method_s == "standard error":
        spread = std / np.sqrt(count)
    elif method_s == "standard deviation":
        spread = std
    else:
        raise ValueError(f"Unsupported method_s: {method_s}")

    return mean, spread
