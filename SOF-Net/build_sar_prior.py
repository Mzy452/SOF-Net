import argparse
import csv

from sofnet.config import load_config
from sofnet.sar_prior import build_prior, write_raster


def split_paths(value):
    return [item.strip() for item in value.split(";") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    prior_config = {**config["sar_prior"]}
    manifest = config["data"]["sar_prior_manifest"]
    with open(manifest, "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        gf3_paths = split_paths(row["gf3"])
        sentinel1_paths = split_paths(row["sentinel1"])
        quality_text = split_paths(row.get("sentinel1_quality", ""))
        quality = [float(value) for value in quality_text] if quality_text else [1.0] * len(sentinel1_paths)
        prior, stack, profile = build_prior(gf3_paths, sentinel1_paths, quality, prior_config)
        write_raster(row["output_prior"], prior, profile)
        write_raster(row["output_stack"], stack, profile)
        print(row["sample_id"], row["output_prior"], row["output_stack"])


if __name__ == "__main__":
    main()

