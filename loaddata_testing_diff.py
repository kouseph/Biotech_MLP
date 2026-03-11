import argparse
import shutil
import subprocess
from pathlib import Path


def copy_for_experiment(root: Path) -> None:
    files_to_copy = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
        "test_fundamentals_available.csv",
        "test_fundamental_recency_days.csv",
    ]
    for name in files_to_copy:
        src = root / name
        if not src.exists():
            print(f"Skipping missing file: {name}")
            continue
        dst = root / f"{src.stem}_diff{src.suffix}"
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare preserved experiment inputs for testing_diff_data.py."
    )
    parser.add_argument(
        "--refresh-upstream",
        action="store_true",
        help="Regenerate base CSVs by running loaddata_j.py first.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    if args.refresh_upstream:
        cmd = ["python", str(root / "loaddata_j.py")]
        print("Running upstream refresh:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=root)

    copy_for_experiment(root)
    print("Done. Use *_diff.csv files with testing_diff_data.py.")


if __name__ == "__main__":
    main()
