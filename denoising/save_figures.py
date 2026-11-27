"""
save_figures.py

FIGURES OF THE EVALUATION OF THE TRANSIENT REMOVAL FILTER (remove_noise)
------------------------------------------------------------------------
Builds the result figures of evaluation_denoise.ipynb (plotly versions, src/evaluation/evaluation_plots.py) from the saved
metrics of the experiments run with a configuration (src/optimization/configs.py) and saves them as .png (scale 5),
.svg and .html.
    - Experiment 1: detection, elimination, conservation, amplitude, factor_analysis
    - Experiment 2: experiment_comparison, interaction_artifacts, mixture_performance, conservation_multiple,
      idempotence

Requires:
    denoising/results/experiments/{config}/experiment_{1, 2}.csv
        Metrics of each mixture (run_experiment.py -c <config>). Experiment 2 also needs experiment_1.csv of the same
        configuration (experiment_comparison)
    denoising/results/optimization/candidates/search/all.json, .../features/search/<group>/all.json
        C1 and C2' (only for the configurations C1 and C2: the figures show REF_INNER and REF_OUTER)

Generates:
    denoising/results/figures/{config}/experiment_{1, 2}/{name}.{png, svg, html}
        One file of each format per figure (wherever the script is run from)

Execution:
    python denoising/save_figures.py [-c <config>] [-e <experiment> ...] [-f <figure> ...] [-s <scale>]
        -c, --config: baseline, C1 or C2 (default: baseline)
        - experiment: 1, 2 or both (default: 1 2)
        -f, --figures: names of the figures to save (default: all the figures of each experiment)
        -s, --scale: scale factor of the .png images (default: 5)
"""


import argparse

from src.evaluation.evaluation_plots import FIGURES, PNG_SCALE, save_figures
from src.optimization.candidate_search import remove_noise_params
from src.optimization.configs import CONFIGS, configuration


def main():
    print()
    parser = argparse.ArgumentParser(description="Save the figures of the evaluation experiments of the transient "
                                                 "removal filter (.png, .svg and .html)")
    parser.add_argument("-c", "--config", choices=CONFIGS, default="baseline",
                        help="Configuration of the constants of remove_noise of the results: baseline, C1 or C2 "
                             "(default: baseline)")
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

    with remove_noise_params(configuration(args.config)):  # Constants shown by the figures (REF_INNER, REF_OUTER)
        for experiment in experiments:
            names = None if args.figures is None else [n for n in args.figures if n in FIGURES[experiment]]
            if names == []:                           # None of the requested figures is of this experiment
                continue
            print(f"Saving the figures of experiment {experiment} ({args.config})...")
            save_figures(experiment, config=args.config, names=names, scale=args.scale)
            print()


if __name__ == "__main__":
    main()
