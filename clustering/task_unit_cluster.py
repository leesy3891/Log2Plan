# task_unit_cluster.py
"""
공유 잠재표현(shared_z_task_unit.npy)을 사용하여 Task Unit을 HDBSCAN 계층 클러스터링
(모델 재학습 없음). **이 파일 안에서** HDBSCAN 파라미터(env/sub)를 자유롭게 조절할 수 있도록
직접 HDBSCAN을 호출하여 2단계 클러스터링을 수행합니다.
출력: task_unit_clustering_results.json
"""
import os
import json
import pickle
import numpy as np
from sklearn.cluster import HDBSCAN


def _save_results_task_unit(cluster_results, out_path):
    # 메타데이터 계산
    total_subclusters = sum(
        len(info["act_des_subclusters"]) for info in cluster_results["env_clusters"].values()
    )
    n_units = sum(info["size"] for info in cluster_results["env_clusters"].values())

    results_with_meta = {
        "metadata": {
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "total_task_units": n_units,
            "total_env_clusters": len(cluster_results["env_clusters"]),
            "total_subclusters": total_subclusters,
            "clustering_method": "hierarchical_hdbscan_shared_z",
        },
        "results": cluster_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_with_meta, f, ensure_ascii=False, indent=2)


def run_task_unit_clustering(
    tu_shared_path: str = "shared_z_task_unit.npy",
    task_unit_vectors_path: str = "task_unit_vectors_legacy.pkl",  # 🔥 legacy 버전 사용
    out_path: str = "task_unit_clustering_results.json",
    # 상위(ENV) 단계 HDBSCAN 파라미터
    env_min_cluster_size: int = 3,
    env_min_samples: int = 2,
    env_metric: str = "cosine",
    env_epsilon: float = 0.3,  # cluster_selection_epsilon
    # 하위(ACT+DES) 단계 HDBSCAN 파라미터
    sub_min_cluster_size: int = 2,
    sub_min_samples: int = 1,
    sub_metric: str = "cosine",
    sub_epsilon: float = 0.2,
):
    if not os.path.exists(tu_shared_path):
        raise FileNotFoundError(f"Missing shared reps: {tu_shared_path}. Run shared_rep.py first.")
    if not os.path.exists(task_unit_vectors_path):
        raise FileNotFoundError(f"Missing vectors: {task_unit_vectors_path}.")

    shared_z = np.load(tu_shared_path)
    with open(task_unit_vectors_path, "rb") as f:
        units = pickle.load(f)

    # 라벨 (출력용)
    env_labels = [u.get("env_text", "unknown") for u in units]
    act_labels = [u.get("act_text", "unknown") for u in units]
    des_labels = [u.get("des_text", "") for u in units]

    # 1단계: ENV 중심 상위 클러스터링 (shared_z 직접 사용)
    print("1️⃣ ENV 기반 상위 HDBSCAN...")
    env_clusterer = HDBSCAN(
        min_cluster_size=max(2, env_min_cluster_size),
        min_samples=env_min_samples,
        metric=env_metric,
        cluster_selection_epsilon=env_epsilon,
    )
    env_clusters = env_clusterer.fit_predict(shared_z)

    unique_env = sorted(set(int(c) for c in env_clusters if c != -1))
    n_noise = int(np.sum(env_clusters == -1))
    print(f"   ENV 클러스터 수: {len(unique_env)}, 노이즈: {n_noise}")

    # 2단계: 각 ENV 클러스터 내 ACT+DES 하위 클러스터링 (shared_z 재사용)
    print("2️⃣ 각 ENV 클러스터 내 ACT+DES HDBSCAN...")
    cluster_results = {"env_clusters": {}, "task_unit_assignments": {}}

    for env_id in unique_env:
        idx = np.where(env_clusters == env_id)[0]
        if len(idx) < 2:
            continue

        sub_shared = shared_z[idx]
        # 충분하면 HDBSCAN, 아니면 단일 클러스터(0)
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

            # Task Unit 목록 구성
            tu_list = []
            for k in sub_idx:
                unit = units[int(k)]
                tu_list.append({
                    "unique_id": unit.get("unique_id", f"TU_{k:04d}"),
                    "title": unit.get("title", ""),
                    "env_tag": unit.get("env_tag", []),
                    "act_tag": unit.get("act_tag", ""),
                    "env_text": env_labels[k],
                    "act_text": act_labels[k],
                    "des_text": des_labels[k],
                    "file_name": unit.get("file_name", ""),
                })
                cluster_results["task_unit_assignments"][tu_list[-1]["unique_id"]] = {
                    "env_cluster": int(env_id),
                    "act_des_cluster": int(sub_id),
                    "cluster_key": f"{int(env_id)}_{int(sub_id)}",
                }

            cluster_results["env_clusters"][int(env_id)]["act_des_subclusters"][int(sub_id)] = {
                "size": int(len(sub_idx)),
                "task_units": tu_list,
            }

        print(
            f"   ENV[{env_id}] → 하위클러스터 {len(cluster_results['env_clusters'][int(env_id)]['act_des_subclusters'])}개"
        )

    _save_results_task_unit(cluster_results, out_path)
    print(f"✅ Task Unit clustering saved to {out_path}")


if __name__ == "__main__":
    run_task_unit_clustering()