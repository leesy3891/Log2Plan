# -*- coding: utf-8 -*-
"""
Enrich task_clustering_results.json with steps from steps_json/tasks_fixed.jsonl.

- Reads tasks_fixed.jsonl (one-JSON-per-line) and builds a map: unique_id -> steps (string)
- Loads task_clustering_results.json and, for every task object under
  results.env_clusters[env_id].act_des_subclusters[sub_id].tasks[*],
  adds a new key "steps" right **after** the existing "content" key (if present),
  using the value from the tasks_fixed map (matched by unique_id).
- Backs up the original clusters file to <path>.bak before writing.

Usage:
  python enrich_steps_into_clustering.py \
      --clusters task_clustering_results.json \
      --tasks ./steps_json/tasks_fixed.jsonl \
      --out task_clustering_results.json        # in-place (writes .bak)

Optional flags:
  --dry_run            : don't write, just print stats
  --overwrite          : overwrite existing task["steps"] if already present (default: False)
  --allow_empty_steps  : still insert steps even if the mapped value is empty string (default: False)

Notes:
  * Matching is done by task["unique_id"]. If not found in tasks_fixed.jsonl map, the task is counted as missing.
  * Output preserves UTF-8 and pretty prints with indent=2 for readability.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import OrderedDict


def load_tasks_steps_map(tasks_jsonl_path: str) -> Dict[str, str]:
    steps_map: Dict[str, str] = {}
    total = 0
    with open(tasks_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = obj.get('unique_id')
            steps = obj.get('steps', '')
            if uid:
                steps_map[uid] = steps if steps is not None else ''
    print(f"📥 Loaded {len(steps_map)}/{total} task lines with unique_id from: {tasks_jsonl_path}")
    return steps_map


def insert_steps_after_content(task: Dict[str, Any], steps_value: str) -> Dict[str, Any]:
    """Return a new OrderedDict placing 'steps' right after 'content' if present.
    If 'content' is absent, returns with 'steps' appended at the end.
    """
    if not isinstance(task, dict):
        return task
    # If already OrderedDict, convert to normal dict to normalize order building
    items = list(task.items())
    new_task = OrderedDict()
    inserted = False
    for k, v in items:
        new_task[k] = v
        if k == 'content' and not inserted:
            new_task['steps'] = steps_value
            inserted = True
    if not inserted:
        # No 'content' key; append at end
        new_task['steps'] = steps_value
    return new_task


def enrich_clusters_with_steps(clusters_path: str, steps_map: Dict[str, str], *, overwrite: bool = False, allow_empty: bool = False) -> Tuple[dict, Dict[str, int]]:
    with open(clusters_path, 'r', encoding='utf-8') as f:
        clusters = json.load(f)

    stats = {
        'total_tasks_seen': 0,
        'steps_added': 0,
        'steps_overwritten': 0,
        'already_had_steps': 0,
        'missing_in_steps_map': 0,
        'empty_steps_skipped': 0,
    }

    results = clusters.get('results', {})
    env_clusters = results.get('env_clusters', {})
    for env_id, env_obj in env_clusters.items():
        act_groups = env_obj.get('act_des_subclusters', {})
        for sub_id, sub_obj in act_groups.items():
            tasks = sub_obj.get('tasks', [])
            if not isinstance(tasks, list):
                continue
            for i, task in enumerate(tasks):
                if not isinstance(task, dict):
                    continue
                stats['total_tasks_seen'] += 1
                uid = task.get('unique_id')
                if not uid:
                    stats['missing_in_steps_map'] += 1
                    continue
                if uid not in steps_map:
                    stats['missing_in_steps_map'] += 1
                    continue
                steps_value = steps_map[uid]
                if (not allow_empty) and (steps_value is None or steps_value == ''):
                    stats['empty_steps_skipped'] += 1
                    continue

                if 'steps' in task and not overwrite:
                    stats['already_had_steps'] += 1
                    continue

                # Place steps after 'content' if present
                new_task = insert_steps_after_content(task, steps_value)

                # Track overwrite/add stats
                if 'steps' in task and overwrite:
                    stats['steps_overwritten'] += 1
                else:
                    stats['steps_added'] += 1

                # Replace in list
                sub_obj['tasks'][i] = new_task

    return clusters, stats


def write_json_with_backup(obj: dict, out_path: str, *, backup_from: str | None = None) -> None:
    out_p = Path(out_path)
    if backup_from:
        # only back up when overwriting original
        bak_p = Path(backup_from + '.bak')
        if out_p.resolve() == Path(backup_from).resolve():
            # in-place overwrite → create .bak
            shutil.copy2(backup_from, bak_p)
            print(f"💾 Backup created: {bak_p}")
    tmp_path = str(out_p) + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)
    print(f"✅ Wrote enriched JSON: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clusters', required=True, help='Path to task_clustering_results.json')
    ap.add_argument('--tasks', required=True, help='Path to steps_json/tasks_fixed.jsonl')
    ap.add_argument('--out', default=None, help='Output path (default: overwrite --clusters in-place)')
    ap.add_argument('--dry_run', action='store_true', help='Do not write output; just print stats')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing task["steps"] if present')
    ap.add_argument('--allow_empty_steps', action='store_true', help='Insert even when steps is empty string')
    args = ap.parse_args()

    steps_map = load_tasks_steps_map(args.tasks)
    enriched, stats = enrich_clusters_with_steps(
        clusters_path=args.clusters,
        steps_map=steps_map,
        overwrite=args.overwrite,
        allow_empty=args.allow_empty_steps,
    )

    print("\n📊 Stats:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

    if args.dry_run:
        print("\n(dry-run) No files written.")
        return

    out_path = args.out or args.clusters
    write_json_with_backup(enriched, out_path, backup_from=(None if args.out else args.clusters))


if __name__ == '__main__':
    main()