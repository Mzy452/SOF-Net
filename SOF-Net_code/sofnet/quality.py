import numpy as np


def combine_quality(indicators, config, valid=None):
    indicators = np.asarray(indicators, dtype=np.float64)
    if indicators.ndim != 3 or indicators.shape[0] != 5:
        raise ValueError('Expected cloud,haze,shadow,sunglint,radiometric indicator rasters')
    valid = np.isfinite(indicators).all(axis=0) & (True if valid is None else np.asarray(valid, bool))
    lower = np.asarray(config['lower'])[:, None, None]
    upper = np.asarray(config['upper'])[:, None, None]
    weights = np.asarray(config['weights'])
    if (upper <= lower).any() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid quality bounds/weights')
    q = np.clip((indicators - lower) / (upper - lower), 0, 1)
    result = np.clip(1 - (q * weights[:, None, None]).sum(axis=0), 0, 1)
    return np.where(valid, result, np.nan).astype('float32'), valid


def patch_quality(quality, valid):
    valid = np.asarray(valid, bool) & np.isfinite(quality)
    return float(np.clip(quality[valid], 0, 1).mean()) if valid.any() else None
