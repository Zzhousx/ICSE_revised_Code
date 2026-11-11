"""Generate STRaP-OS sensitivity rankings from the benchmark inputs."""
import argparse
from pathlib import Path
import subprocess
import sys

CONFIGURATIONS = (
    'semantic_coverage', 'time_equivalent', 'distance_30', 'distance_50',
    'distance_100', 'pedestrian_proxy', 'joined_recordings', 'single_pass',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=int, choices=(1, 2, 3))
    parser.add_argument('--config', choices=CONFIGURATIONS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    datasets = (args.dataset,) if args.dataset is not None else (1, 2, 3)
    configs = (args.config,) if args.config is not None else CONFIGURATIONS
    for config in configs:
        for dataset in datasets:
            subprocess.run([
                sys.executable, str(root / 'strap_os' / 'launch_fullrank.py'),
                '--config', config, '--dataset', str(dataset),
                '--kept-interval', 'half_open',
            ], cwd=root, check=True)


if __name__ == '__main__':
    main()
