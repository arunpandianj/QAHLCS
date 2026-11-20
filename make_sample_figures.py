"""Write the sample-data figures. Run after the main stage so that the prediction examples can be drawn.

    python make_sample_figures.py --results results
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=None); ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
if a.quick: os.environ["QPD_QUICK"] = "1"
if a.results: os.environ["QPD_RESULTS"] = a.results

from qpd.sample_figures import dataset_samples, prediction_examples  # noqa: E402

if __name__ == "__main__":
    dataset_samples(); prediction_examples()
