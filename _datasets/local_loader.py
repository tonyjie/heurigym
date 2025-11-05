# _datasets/local_loader.py
import os, json, glob
from pathlib import Path
from typing import Dict, List, Optional, Iterable


# --------- 轻量容器（兼容 len(ds)、ds[i]、for ex in ds）---------
class LocalDataset:
    def __init__(self, records: List[Dict]): self._data = records
    def __len__(self) -> int: return len(self._data)
    def __getitem__(self, i: int) -> Dict: return self._data[i]
    def __iter__(self): return iter(self._data)


# ===================== 通用工具 =====================
def _repo_datasets_root() -> Path:
    # 本文件: .../heurigym/_datasets/local_loader.py
    # 数据根: .../heurigym/_datasets
    return Path(__file__).resolve().parent

def _abs(p: os.PathLike | str) -> str:
    return str(Path(p).resolve())

def _normalize_record(raw: Dict) -> Dict:
    # 若字段名已符合 HeuriGym 其它代码需求（如 prompt/solver 会直接读取），则原样返回
    return raw


# ===================== operator_scheduling =====================
def load_local_operator_scheduling(
    root: str,
    split: str = "train",
    patterns: Optional[List[str]] = None,
) -> LocalDataset:
    """
    从本地目录加载 operator_scheduling 数据。
    默认结构：{root}/operator_scheduling/dataset/{split}/*.jsonl|json
    也支持 root 直接指到 split 目录。
    """
    if patterns is None:
        patterns = ["*.jsonl", "*.json"]

    split_dir_a = os.path.join(root, "operator_scheduling", "dataset", split)
    base_dir = split_dir_a if os.path.isdir(split_dir_a) else root

    files: List[str] = []
    for pat in patterns:
        files += glob.glob(os.path.join(base_dir, pat))

    if not files:
        raise FileNotFoundError(f"No data files under: {base_dir}")

    records: List[Dict] = []
    for fp in sorted(files):
        fp = _abs(fp)  # 统一成绝对路径（将来谁用都不踩相对路径坑）
        if fp.endswith(".jsonl"):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        records.append(_normalize_record(json.loads(s)))
        elif fp.endswith(".json"):
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    records += [_normalize_record(x) for x in obj]
                else:
                    records.append(_normalize_record(obj))
    return LocalDataset(records)


# ===================== SGD / s_gd2（SuiteSparse .mat） =====================
def _list_mat_files(problem_name: str, split: str = "demo", root_override: Optional[str] = None) -> List[str]:
    """
    列出 {root}/{problem_name}/{split}/*.mat，返回**绝对路径**列表。
    root_override 为空时默认用仓库内 _datasets。
    """
    base = Path(root_override).resolve() if root_override else _repo_datasets_root()
    root = base / problem_name / split
    if not root.is_dir():
        raise FileNotFoundError(f"Local dataset split not found: {root}")
    files = sorted(root.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files under: {root}")
    return [str(p.resolve()) for p in files]


def load_local_sgd_like(problem_name: str, split: str = "demo", root_override: Optional[str] = None) -> LocalDataset:
    mats = _list_mat_files(problem_name, split, root_override)
    records = []
    for p in mats:
        name = os.path.splitext(os.path.basename(p))[0]
        records.append({"name": name, "path": p})
    return LocalDataset(records)


# ===================== 统一入口 =====================
def load_local_dataset(
    problem_name: str,
    split: str = "demo",
    root_override: Optional[str] = None,
):
    root = _abs(root_override) if root_override else str(_repo_datasets_root())

    if problem_name == "operator_scheduling":
        return load_local_operator_scheduling(root, split)

    # 把 sgd_graph_layout 也归入 sgd 类
    if problem_name in {"SGD", "s_gd2", "sgd_graph_layout"}:
        return load_local_sgd_like(problem_name, split, root)

    raise ValueError(f"Unsupported local problem: {problem_name}")


# =============== 便捷 CLI（自测用，可不理会） ===============
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, help="operator_scheduling | SGD | s_gd2")
    ap.add_argument("--split", default="demo")
    ap.add_argument("--root", default=None, help="override dataset root (optional)")
    args = ap.parse_args()

    ds = load_local_dataset(args.problem, args.split, args.root)
    print("len:", len(ds))
    first = next(iter(ds))
    print("sample:", first if isinstance(first, dict) else type(first))
    if isinstance(first, dict) and "path" in first:
        print("isabs:", os.path.isabs(first["path"]))
