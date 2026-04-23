#!/usr/bin/env python3

import ast
import hashlib
import os
from collections import defaultdict


def iter_python_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def collect_functions(root="."):
    functions = []
    for path in iter_python_files(root):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                src = handle.read()
            tree = ast.parse(src, filename=path)
        except Exception:
            continue
        lines = src.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", node.lineno)
                body = "\n".join(lines[start - 1 : end]).strip()
                normalized = "\n".join(line.strip() for line in body.splitlines())
                functions.append(
                    {
                        "name": node.name,
                        "path": path,
                        "lineno": start,
                        "hash": hashlib.sha256(normalized.encode()).hexdigest()[:16],
                    }
                )
    return functions


def main():
    functions = collect_functions()
    by_name = defaultdict(list)
    by_hash = defaultdict(list)
    for function in functions:
        by_name[function["name"]].append(function)
        by_hash[function["hash"]].append(function)

    print("Top duplicate names")
    for name, items in sorted(by_name.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:50]:
        if len(items) < 2:
            continue
        print(f"{len(items):3d} {name}")
        for item in items[:8]:
            print(f"    - {item['path']}:{item['lineno']}")

    print()
    print("Top exact duplicate groups")
    groups = [items for items in by_hash.values() if len(items) > 1]
    for items in sorted(groups, key=lambda group: (-len(group), group[0]["name"]))[:50]:
        print(f"{len(items):3d} {items[0]['name']}")
        for item in items[:8]:
            print(f"    - {item['path']}:{item['lineno']}")


if __name__ == "__main__":
    main()
