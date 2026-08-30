from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


def local_standardize(image, window, epsilon):
    mean = uniform_filter(image, size=window, mode="reflect")
    mean_square = uniform_filter(image * image, size=window, mode="reflect")
    variance = np.maximum(mean_square - mean * mean, 0.0)
    return (image - mean) / (np.sqrt(variance) + epsilon)


def cfar_confidence(image_db, background_window, guard_window, pfa, epsilon):
    intensity = np.power(10.0, image_db / 10.0)
    background_count = background_window * background_window
    guard_count = guard_window * guard_window
    sample_count = background_count - guard_count
    background_sum = uniform_filter(intensity, size=background_window, mode="reflect") * background_count
    guard_sum = uniform_filter(intensity, size=guard_window, mode="reflect") * guard_count
    background_mean = (background_sum - guard_sum) / max(sample_count, 1)
    alpha = sample_count * (np.power(pfa, -1.0 / max(sample_count, 1)) - 1.0)
    threshold = alpha * background_mean
    confidence = (intensity - threshold) / (np.abs(threshold) + epsilon)
    return np.clip(confidence, 0.0, 1.0)


def read_multiband(path):
    with rasterio.open(path) as source:
        array = source.read().astype(np.float32)
        profile = source.profile.copy()
    return array, profile


def sensor_evidence(paths, config):
    if not paths:
        return None
    date_confidences = []
    date_means = []
    date_z = []
    profile = None
    for path in paths:
        array, current_profile = read_multiband(path)
        if profile is None:
            profile = current_profile
        z_channels = np.stack(
            [local_standardize(channel, config["background_window"], config["epsilon"]) for channel in array]
        )
        cfar_channels = np.stack(
            [
                cfar_confidence(
                    channel,
                    config["background_window"],
                    config["guard_window"],
                    config["pfa"],
                    config["epsilon"],
                )
                for channel in array
            ]
        )
        date_means.append(np.mean(array, axis=0))
        date_z.append(np.max(z_channels, axis=0))
        date_confidences.append(np.max(cfar_channels, axis=0))
    return (
        np.stack(date_means),
        np.stack(date_z),
        np.stack(date_confidences),
        profile,
    )


def build_prior(gf3_paths, sentinel1_paths, quality, config):
    gf3 = sensor_evidence(gf3_paths, config)
    sentinel1 = sensor_evidence(sentinel1_paths, config)
    if gf3 is None and sentinel1 is None:
        raise ValueError("At least one SAR source is required")
    profile = gf3[3] if gf3 is not None else sentinel1[3]
    height = profile["height"]
    width = profile["width"]
    if gf3 is None:
        gf3_mean = np.zeros((1, height, width), dtype=np.float32)
        gf3_z = np.zeros_like(gf3_mean)
        gf3_surface = np.zeros((height, width), dtype=np.float32)
        gf3_weight = 0.0
    else:
        gf3_mean, gf3_z, gf3_confidence, _ = gf3
        gf3_surface = np.max(gf3_confidence, axis=0)
        gf3_weight = config["gf3_weight"]
    if sentinel1 is None:
        s1_mean = np.zeros((1, height, width), dtype=np.float32)
        s1_z = np.zeros_like(s1_mean)
        persistence = np.zeros((height, width), dtype=np.float32)
        sentinel1_weight = 0.0
    else:
        s1_mean, s1_z, s1_confidence, _ = sentinel1
        if len(quality) != s1_confidence.shape[0]:
            raise ValueError("Sentinel-1 quality weights must match the number of scenes")
        quality = np.asarray(quality, dtype=np.float32)
        quality = quality / max(float(quality.sum()), config["epsilon"])
        persistence = np.sum(s1_confidence * quality[:, None, None], axis=0)
        sentinel1_weight = config["sentinel1_weight"]
    weight_sum = gf3_weight + sentinel1_weight
    gf3_weight = gf3_weight / weight_sum
    sentinel1_weight = sentinel1_weight / weight_sum
    prior = gf3_weight * gf3_surface + sentinel1_weight * persistence
    prior = np.clip(prior, 0.0, 1.0)
    stack = np.stack(
        [
            np.mean(gf3_mean, axis=0),
            np.max(gf3_z, axis=0),
            gf3_surface,
            np.mean(s1_mean, axis=0),
            np.max(s1_z, axis=0),
            persistence,
            (persistence >= config["persistence_threshold"]).astype(np.float32),
            prior,
        ]
    ).astype(np.float32)
    return prior.astype(np.float32), stack, profile


def write_raster(path, array, profile):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = array[None] if array.ndim == 2 else array
    output_profile = profile.copy()
    output_profile.update(count=data.shape[0], dtype="float32", compress="deflate")
    with rasterio.open(destination, "w", **output_profile) as target:
        target.write(data.astype(np.float32))
