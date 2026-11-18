"""
save_figures.py

FIGURES OF THE EVALUATION OF THE TRANSIENT REMOVAL FILTER (remove_noise)
------------------------------------------------------------------------
Builds the result figures of evaluation_denoise.ipynb (plotly versions, src/evaluation_plots.py) from the saved
metrics of the experiments and saves them as .png (scale 5), .svg and .html.
    - Experiment 1: detection, elimination, conservation, amplitude, factor_analysis
    - Experiment 2: experiment_comparison, interaction_artifacts, mixture_performance, conservation_multiple,
      idempotence

Requires:
    denoising/results/experiments/experiment_{1, 2}.csv
        Metrics of each mixture (run_experiment.py). Experiment 2 also needs experiment_1.csv (experiment_comparison)

Generates:
    denoising/results/figures/experiment_{1, 2}/{name}.{png, svg, html}
        One file of each format per figure (wherever the script is run from)

Execution:
    python denoising/save_figures.py [-e <experiment> ...] [-f <figure> ...] [-s <scale>]
        - experiment: 1, 2 or both (default: 1 2)
        -f, --figures: names of the figures to save (default: all the figures of each experiment)
        -s, --scale: scale factor of the .png images (default: 5)
"""


import argparse

from src.evaluation_plots import FIGURES, PNG_SCALE, save_figures


def main():
    print()
    parser = argparse.ArgumentParser(description="Save the figures of the evaluation experiments of the transient "
                                                 "removal filter (.png, .svg and .html)")
    parser.add_argument("-e", "--experiment", nargs="+", type=int, choices=sorted(FIGURES), default=sorted(FIGURES),
                        help="Experiment(s) whose figures are saved: 1, 2 or both (default: 1 2)")
    parser.add_argument("-f", "--figures", nargs="+", default=None,
                        help="Figures to save (default: all). Experiment 1: " + ", ".join(FIGURES[1]) +
                             ". Experiment 2: " + ", ".join(FIGURES[2]))
    parser.add_argument("-s", "--scale", type=float, default=PNG_SCALE,
                        help=f"Scale factor of the .png images (default: {PNG_SCALE})")
    args = parser.parse_args()

    experiments = sorted(set(args.experiment))
    if args.figures is not None:                      # Every name must belong to one of the selected experiments
        valid = {name for e in experiments for name in FIGURES[e]}
        unknown = [name for name in args.figures if name not in valid]
        if unknown:
            parser.error(f"unknown figures for experiment(s) {experiments}: {unknown}")

    for experiment in experiments:
        names = None if args.figures is None else [n for n in args.figures if n in FIGURES[experiment]]
        if names == []:                               # None of the requested figures is of this experiment
            continue
        print(f"Saving the figures of experiment {experiment}...")
        save_figures(experiment, names=names, scale=args.scale)
        print()


if __name__ == "__main__":
    main()
