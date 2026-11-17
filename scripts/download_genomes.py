#!/usr/bin/env python3
"""Thin wrapper: `python scripts/download_genomes.py --manifest ... --outdir genomes`.
Equivalent to `mss download-genomes`. Requires internet and the NCBI Datasets CLI."""
import sys

from mss.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["download-genomes", *sys.argv[1:]]))
