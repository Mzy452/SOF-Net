from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


def _sum(a, window):
    return uniform_filter(a.astype(np.float64), size=window, mode='constant', cval=0) * window ** 2


def local_standardize(image, window, epsilon):
    valid = np.isfinite(image)
    safe = np.where(valid, image, 0)
    count = _sum(valid, window)
    mean = _sum(safe, window) / np.maximum(count, 1)
    variance = np.maximum(_sum(safe * safe, window) / np.maximum(count, 1) - mean ** 2, 0)
    return np.where(valid & (count > 0), (safe - mean) / (np.sqrt(variance) + epsilon), np.nan)


def cfar_confidence(image_db, background_window, guard_window, pfa, epsilon, upper_factor=8.0):
    if not (background_window > guard_window >= 1 and background_window % 2 and guard_window % 2):
        raise ValueError('CFAR requires odd background > guard windows')
    if not 0 < pfa < 1 or upper_factor <= 1:
        raise ValueError('Invalid CFAR probability or upper factor')
    valid = np.isfinite(image_db)
    intensity = np.where(valid, np.power(10., np.where(valid, image_db, 0) / 10.), 0)
    count = np.maximum(np.rint(_sum(valid, background_window) - _sum(valid, guard_window)), 0)
    total = _sum(intensity, background_window) - _sum(intensity, guard_window)
    mean = np.maximum(total, 0) / np.maximum(count, 1)
    alpha = count * np.expm1(-np.log(pfa) / np.maximum(count, 1))
    threshold = alpha * mean
    confidence = np.clip((intensity - threshold) / np.maximum((upper_factor - 1) * threshold, epsilon), 0, 1)
    return np.where(valid & (count > 0), confidence, np.nan).astype(np.float32)


def read_multiband(path):
    with rasterio.open(path) as source:
        return source.read(masked=True).astype('float32').filled(np.nan), source.profile.copy()


def _check_grid(a, b):
    for key in ('width', 'height', 'transform', 'crs'):
        if a[key] != b[key]:
            raise ValueError(f'SAR grid differs in {key}; register before prior construction')


def _max(a, axis=0):
    valid = np.isfinite(a)
    return np.where(valid.any(axis=axis), np.where(valid, a, -np.inf).max(axis=axis), np.nan)


def _mean(a, weights=None):
    valid = np.isfinite(a)
    weights = np.ones(a.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (a.shape[0],) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative and match acquisitions, with positive sum')
    w = weights[:, None, None] * valid
    denominator = w.sum(axis=0)
    numerator = (np.where(valid, a, 0) * w).sum(axis=0)
    return np.divide(numerator, denominator, out=np.full(a.shape[1:], np.nan), where=denominator > 0)


def sensor_evidence(paths, config):
    if not paths:
        return None
    means, anomalies, confidences = [], [], []
    profile = None
    for path in paths:
        array, current = read_multiband(path)
        if profile is not None:
            _check_grid(profile, current)
        profile = current
        z = np.stack([local_standardize(c, config['background_window'], config['epsilon']) for c in array])
        cf = np.stack([cfar_confidence(c, config['background_window'], config['guard_window'], config['pfa'],
                                      config['epsilon'], config.get('upper_threshold_factor', 8.)) for c in array])
        means.append(_mean(array))
        anomalies.append(_max(z))
        confidences.append(_max(cf))
    return np.stack(means), np.stack(anomalies), np.stack(confidences), profile


def build_prior(gf3_paths, sentinel1_paths, quality, config):
    gf3, s1 = sensor_evidence(gf3_paths, config), sensor_evidence(sentinel1_paths, config)
    if gf3 is None and s1 is None:
        raise ValueError('At least one SAR product is required')
    if gf3 is not None and s1 is not None:
        _check_grid(gf3[3], s1[3])
    profile = (gf3 if gf3 is not None else s1)[3]
    shape = (profile['height'], profile['width'])
    missing = np.full(shape, np.nan)
    g = _max(gf3[2]) if gf3 is not None else missing
    p = _mean(s1[2], quality if len(quality) else None) if s1 is not None else missing
    gv, sv = np.isfinite(g), np.isfinite(p)
    gw, sw = float(config['gf3_weight']), float(config['sentinel1_weight'])
    if gw <= 0 or sw <= 0:
        raise ValueError('Sensor weights must be positive; missing coverage is handled per pixel')
    denominator = gw * gv + sw * sv
    prior = np.divide(gw * np.nan_to_num(g) + sw * np.nan_to_num(p), denominator,
                      out=np.full(shape, np.nan), where=denominator > 0)
    prior = np.clip(prior, 0, 1)
    stack = np.stack([_mean(gf3[0]) if gf3 else missing, _max(gf3[1]) if gf3 else missing, g,
                      _mean(s1[0]) if s1 else missing, _max(s1[1]) if s1 else missing, p,
                      (p >= config['persistence_threshold']).astype(float), prior])
    stack = np.nan_to_num(stack).astype('float32')
    stack[:, ~(gv | sv)] = np.nan
    return prior.astype('float32'), stack, profile


def write_raster(path, array, profile):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = array[None] if array.ndim == 2 else array
    output_profile = profile.copy()
    output_profile.update(driver='GTiff', count=data.shape[0], dtype='float32', nodata=np.nan, compress='deflate')
    with rasterio.open(destination, 'w', **output_profile) as target:
        target.write(data.astype('float32'))
