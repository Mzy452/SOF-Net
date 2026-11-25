import argparse
from sofnet.config import load_config
from sofnet.quality import combine_quality
from sofnet.sar_prior import read_multiband, write_raster


def main():
    p = argparse.ArgumentParser(description='Combine five externally calculated indicators')
    p.add_argument('--config', required=True)
    p.add_argument('--indicators', required=True, help='Aligned five-band indicator GeoTIFF')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    raw, profile = read_multiband(a.indicators)
    quality, _ = combine_quality(raw, load_config(a.config)['quality'])
    write_raster(a.output, quality, profile)


if __name__ == '__main__':
    main()
