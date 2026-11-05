"""Fix the train / val / test split BEFORE you take photos or train anything.

    python -m dataset.make_split --count 100 --seed 42 --out dataset/splits

Writes planned file names (img_001.jpg ...) into train.txt, val.txt and test.txt (70 / 20 / 10) plus all.txt
(capture order). Photograph in the order of all.txt: your i-th photo becomes img_i. Commit these lists first.
Note: shots taken one after another look alike, so reposition the object between shots; otherwise a
test image can be a near copy of a training image and the test metrics will look better than they are.
"""
import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="dataset/splits")
    args = parser.parse_args()

    names = [f"img_{i:03d}.jpg" for i in range(1, args.count + 1)]
    shuffled = names[:]
    random.Random(args.seed).shuffle(shuffled)
    n_train, n_val = round(args.count * 0.7), round(args.count * 0.2)
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "all.txt").write_text("\n".join(names) + "\n")
    for split, files in splits.items():
        (out / f"{split}.txt").write_text("\n".join(sorted(files)) + "\n")
        print(f"{split}: {len(files)} images")


if __name__ == "__main__":
    main()
