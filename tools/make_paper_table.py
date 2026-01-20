import pandas as pd
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "outputs" / "model_comparison.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing comparison CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure order A0, A1, A3, then any others
    order = ["A0_XGBoost", "A1_MA_ARIMA", "A3_Hybrid"]
    df["__order"] = df["Model"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("__order").drop(columns="__order")

    # Round numeric columns
    for col in ["MAE", "RMSE", "Peak_RMSE"]:
        if col in df.columns:
            df[col] = df[col].astype(float).round(2)

    md_lines = [
        "| Model | MAE (MW) | RMSE (MW) | Peak_RMSE 18:00-22:00 (MW) | Role |",
        "|---|---:|---:|---:|---|",
    ]

    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['Model']} | {r['MAE']} | {r['RMSE']} | {r['Peak_RMSE']} | {r.get('Role', '')} |"
        )

    paper_md = "\n".join(md_lines)

    out_md = root / "outputs" / "paper_table.md"
    out_md.write_text(paper_md, encoding="utf-8")

    print(f"Wrote: {out_md}")
    print("\n" + paper_md)


if __name__ == "__main__":
    main()
