"""What does a refusal threshold cost? Apply thresholds to measurement/results/accuracy.csv after the fact.

    python -m measurement.threshold_sweep --column confidence --values 0.5 0.7 0.8 0.9 0.95
    python -m measurement.threshold_sweep --column skew --values 0.005 0.01 0.02 0.03 0.05

confidence: keep rows with value >= threshold.  skew: keep rows with value <= threshold.
Uses the undistorted measurements. Prints how many images are refused and the error of those kept.
"""
import argparse
import csv
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="measurement/results/accuracy.csv")
    parser.add_argument("--column", choices=["confidence", "skew"], required=True)
    parser.add_argument("--values", type=float, nargs="+", required=True)
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv)) if r["und_status"] == "ok"]
    for r in rows:
        r["value"] = float(r[f"und_{args.column}"])
        r["errors"] = [abs(float(r["und_width_mm"]) - float(r["true_width_mm"])),
                       abs(float(r["und_height_mm"]) - float(r["true_height_mm"]))]

    print(f"{'threshold':>10}{'accepted':>10}{'refused':>9}{'MAE mm':>9}{'worst mm':>10}")
    for threshold in args.values:
        kept = [r for r in rows if (r["value"] >= threshold if args.column == "confidence" else r["value"] <= threshold)]
        errors = [e for r in kept for e in r["errors"]]
        mae = f"{statistics.mean(errors):.3f}" if errors else "-"
        worst = f"{max(errors):.3f}" if errors else "-"
        print(f"{threshold:>10}{len(kept):>10}{len(rows) - len(kept):>9}{mae:>9}{worst:>10}")


if __name__ == "__main__":
    main()
