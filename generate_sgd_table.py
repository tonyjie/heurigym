#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path
from datetime import datetime   # ← 新增

# 固定的 SGD baseline stress（timeout=120s）
# 就是你表里这一列的值
SGD_BASELINE = {
    "1138_bus": 39944.95351,
    "bcspwr07": 59395.84885,
    "commanche_dual": 5641934.021,
    "delaunay_n10": 37254.213,
    "karate": 37.91321,
    "power": 1668972.397,
    "qh882": 22054.24163,
    "shuttle_eddy": 4566858.118,
    "adjnoun": 830.0,
    "celegansneural": 5363.34014,
    "chesapeake": 87.09515,
    "dolphins": 114.00718,
}


def generate_results_table(input_dir, output_csv, baseline_title):
    """
    从若干 run_summary.json 生成一个总表 CSV。
    - baseline 列用上面写死的 SGD_BASELINE
    - geometric mean 按 stress / nodes^2 计算（只算 eval 行）
    """
    input_dir = Path(input_dir)

    # datasets: name -> { "nodes": ..., "type": ..., "baseline": ..., "<model> iter0": cost, ... }
    datasets = {}

    # 每一列（model+iter）的 meta 信息，用来写 remarks 行
    # col_key -> { model, iter, timeout, temperature, run_id }
    col_meta = {}

    # 最终列顺序
    col_order = []

    # ---------- 1. 扫 input_dir 中的所有 run_summary.json ----------
    for path in sorted(input_dir.glob("*.json")):
        try:
            summary = json.loads(path.read_text())
        except Exception as e:
            print(f"[warn] skip {path}: {e}")
            continue

        model = summary.get("model", path.stem)
        run_id = summary.get("run_id", "")
        config = summary.get("config", {}) or {}
        timeout = config.get("timeout(s)") or summary.get("timeout")
        temperature = config.get("temperature") or summary.get("temperature")

        datasets_list = summary.get("datasets", [])
        if not isinstance(datasets_list, list):
            print(f"[warn] no datasets list in {path}")
            continue

        for ds in datasets_list:
            name = ds.get("name")
            if not name:
                continue

            nodes = ds.get("nodes")
            ds_type = ds.get("type", "eval")
            iter_costs = ds.get("iteration_costs", {}) or {}

            row = datasets.setdefault(name, {"nodes": nodes, "type": ds_type})

            # nodes / type：优先第一次
            if row.get("nodes") is None and nodes is not None:
                row["nodes"] = nodes
            if row.get("type") is None and ds_type is not None:
                row["type"] = ds_type

            # baseline：直接用固定表
            if name in SGD_BASELINE:
                row["baseline"] = SGD_BASELINE[name]

            # 每个 iteration 生成一列 key："model iter0"
            for iter_id_str, cost in iter_costs.items():
                col_key = f"{model} iter{iter_id_str}"

                if col_key not in col_order:
                    col_order.append(col_key)
                    col_meta[col_key] = {
                        "model": model,
                        "iter": iter_id_str,
                        "timeout": timeout,
                        "temperature": temperature,
                        "run_id": run_id,
                        "source_file": str(path),
                    }

                try:
                    cost_f = float(cost)
                except (TypeError, ValueError):
                    continue
                row[col_key] = cost_f

    if not datasets:
        print("[error] no dataset data found, check your input_dir")
        return

    # ---------- 2. dataset 排序：先 eval 再 demo，然后按名字 ----------
    def sort_key(item):
        name, row = item
        t = row.get("type", "eval")
        return (0 if t == "eval" else 1, name)

    sorted_rows = sorted(datasets.items(), key=sort_key)

    # ---------- 3. 写 CSV ----------
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = ["Dataset", "Nodes", "Type", baseline_title]
    header.extend(col_order)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3.1 表头
        writer.writerow(header)

        # 3.2 每个 dataset 一行
        def fmt(x):
            if isinstance(x, (int, float)):
                return f"{x:.6f}"
            return "" if x is None else str(x)

        for name, row in sorted_rows:
            nodes = row.get("nodes", "")
            ds_type = row.get("type", "")
            baseline = row.get("baseline", "")

            line = [
                name,
                nodes if nodes is not None else "",
                ds_type,
                fmt(baseline),
            ]

            for col in col_order:
                line.append(fmt(row.get(col, "")))

            writer.writerow(line)

        # ---------- 3.3 geometric mean：按 stress / nodes^2 ----------
        gm_row = ["geometric mean:", "", ""]

        def gm_for_column(value_getter):
            vals = []
            for _, row in sorted_rows:
                if row.get("type", "eval") != "eval":
                    continue
                nodes = row.get("nodes")
                if not isinstance(nodes, (int, float)) or nodes <= 0:
                    continue
                val = value_getter(row)
                if not isinstance(val, (int, float)) or val <= 0:
                    continue
                x = val / (nodes * nodes)
                vals.append(x)
            if not vals:
                return ""
            gm = math.exp(sum(math.log(v) for v in vals) / len(vals))
            return f"{gm:.11f}"

        # baseline 列 GM
        gm_row.append(gm_for_column(lambda r: r.get("baseline")))

        # 各模型列 GM
        for col in col_order:
            gm_row.append(gm_for_column(lambda r, c=col: r.get(c)))

        writer.writerow(gm_row)

        # ---------- 3.4 remarks 行 ----------
        remarks_row = ["remarks:", "", ""]

        # baseline 那列先留空
        remarks_row.append("")

        for col in col_order:
            meta = col_meta.get(col, {})
            pieces = []
            if meta.get("timeout") is not None:
                pieces.append(f"timeout={meta['timeout']}")
            if meta.get("temperature") is not None:
                pieces.append(f"temperature={meta['temperature']}")
            cell = "\n".join(pieces) if pieces else ""
            remarks_row.append(cell)

        writer.writerow(remarks_row)

    print(f"[info] table written to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="放 run_summary.json 的目录（你挑好的那几次 run 的 json 丢这里）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sgd_results_table.csv",
        help="输出 CSV 文件路径（基础名，会自动加时间戳）",
    )
    parser.add_argument(
        "--baseline-title",
        type=str,
        default="SGD baseline (timeout=120s)",
        help="baseline 列标题（会直接出现在表头）",
    )
    args = parser.parse_args()

    # 在这里加时间戳，保证不会覆盖旧文件
    base_path = Path(args.output)
    base = base_path.with_suffix("")          # 去掉原来的后缀（不管你写不写 .csv）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = base.parent / f"{base.name}_{ts}.csv"

    generate_results_table(
        args.input_dir,
        str(final_path),
        args.baseline_title,
    )


if __name__ == "__main__":
    main()
