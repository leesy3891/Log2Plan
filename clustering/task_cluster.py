# task_cluster.py
"""
개별 Task도 ENV/ACT/DES로부터 **동일한 공유 잠재표현**(shared_z_task.npy)을 사용해
Task Unit과 동일한 계층 HDBSCAN으로 클러스터링합니다.
이 파일에서 HDBSCAN 파라미터(env/sub)를 자유롭게 수정할 수 있습니다.
출력: task_clustering_results.json (구조는 Task Unit 결과와 호환)
"""
import os
import json
import pickle
import numpy as np
from sklearn.cluster import HDBSCAN


def _save_results_task(cluster_results, out_path):
    total_subclusters = sum(
        len(info["act_des_subclusters"]) for info in cluster_results["env_clusters"].values()
    )
    n_tasks = sum(info["size"] for info in cluster_results["env_clusters"].values())

    results_with_meta = {
        "metadata": {
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "total_tasks": n_tasks,
            "total_env_clusters": len(cluster_results["env_clusters"]),
            "total_subclusters": total_subclusters,
            "noise_tasks": 0,  # 필요 시 계산 가능
            "clustering_method": "hierarchical_hdbscan_shared_z_tasks",
        },
        "results": cluster_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_with_meta, f, ensure_ascii=False, indent=2)


def _as_unit_like(task):
    return {
        "unique_id": task.get("unique_id", task.get("id", "")),
        "title": task.get("desc_text", ""),
        "env_tag": [task.get("env_text", "unknown")],
        "act_tag": task.get("act_text", "unknown"),
        "env_text": task.get("env_text", "unknown"),
        "act_text": task.get("act_text", "unknown"),
        "des_text": task.get("desc_text", ""),
        "file_name": task.get("file_name", ""),
        "parent_unit_id": task.get("parent_unit_id", ""),
        "parent_unit_name": task.get("parent_unit_name", ""),
    }


essential_keys = [
    "unique_id",
    "env_text",
    "act_text",
    "des_text",
    "env_tag",
    "act_tag",
    "file_name",
]


def run_task_clustering(
    task_shared_path: str = "shared_z_task.npy",
    task_vectors_path: str = "task_vectors_legacy.pkl",           # 🔥 legacy 버전 사용
    out_path: str = "task_clustering_results.json",
    # 상위(ENV) 단계 HDBSCAN 파라미터
    env_min_cluster_size: int = 3,
    env_min_samples: int = 3,
    env_metric: str = "cosine",
    env_epsilon: float = 0.15,
    # 하위(ACT+DES) 단계 HDBSCAN 파라미터
    sub_min_cluster_size: int = 2,
    sub_min_samples: int = 2,
    sub_metric: str = "cosine",
    sub_epsilon: float = 0.08,
):
    if not os.path.exists(task_shared_path):
        raise FileNotFoundError(f"Missing shared reps: {task_shared_path}. Run shared_rep.py first.")
    if not os.path.exists(task_vectors_path):
        raise FileNotFoundError(f"Missing vectors: {task_vectors_path}.")

    shared_z = np.load(task_shared_path)
    with open(task_vectors_path, "rb") as f:
        tasks = pickle.load(f)

    env_labels = [t.get("env_text", "unknown") for t in tasks]
    act_labels = [t.get("act_text", "unknown") for t in tasks]
    des_labels = [t.get("desc_text", "") for t in tasks]

    # 1단계: ENV 중심 상위 클러스터링
    print("1️⃣ ENV 기반 상위 HDBSCAN (Task)...")
    env_clusterer = HDBSCAN(
        min_cluster_size=max(3, env_min_cluster_size),
        min_samples=env_min_samples,
        metric=env_metric,
        cluster_selection_epsilon=env_epsilon,
    )
    env_clusters = env_clusterer.fit_predict(shared_z)

    unique_env = sorted(set(int(c) for c in env_clusters if c != -1))
    n_noise = int(np.sum(env_clusters == -1))
    print(f"   ENV 클러스터 수: {len(unique_env)}, 노이즈: {n_noise}")

    # 2단계: 각 ENV 클러스터 내 ACT+DES
    print("2️⃣ 각 ENV 클러스터 내 ACT+DES HDBSCAN (Task)...")
    cluster_results = {"env_clusters": {}, "task_assignments": {}}

    for env_id in unique_env:
        idx = np.where(env_clusters == env_id)[0]
        if len(idx) < 2:
            continue

        sub_shared = shared_z[idx]
        if len(idx) >= max(2, sub_min_cluster_size):
            sub_clusterer = HDBSCAN(
                min_cluster_size=max(2, sub_min_cluster_size),
                min_samples=sub_min_samples,
                metric=sub_metric,
                cluster_selection_epsilon=sub_epsilon,
            )
            sub_clusters = sub_clusterer.fit_predict(sub_shared)
        else:
            sub_clusters = np.zeros(len(idx), dtype=int)

        cluster_results["env_clusters"][int(env_id)] = {
            "size": int(len(idx)),
            "act_des_subclusters": {},
        }

        for sub_id in sorted(set(int(c) for c in sub_clusters if c != -1)):
            sub_idx = idx[sub_clusters == sub_id]

            t_list = []
            for k in sub_idx:
                t = tasks[int(k)]
                unit_like = _as_unit_like(t)
                t_list.append(unit_like)
                cluster_results["task_assignments"][unit_like["unique_id"]] = {
                    "env_cluster": int(env_id),
                    "act_des_cluster": int(sub_id),
                    "cluster_key": f"{int(env_id)}_{int(sub_id)}",
                }

            cluster_results["env_clusters"][int(env_id)]["act_des_subclusters"][int(sub_id)] = {
                "size": int(len(sub_idx)),
                "tasks": t_list,
            }

        print(
            f"   ENV[{env_id}] → 하위클러스터 {len(cluster_results['env_clusters'][int(env_id)]['act_des_subclusters'])}개"
        )

    _save_results_task(cluster_results, out_path)
    print(f"✅ Task clustering saved to {out_path}")


if __name__ == "__main__":
    run_task_clustering()