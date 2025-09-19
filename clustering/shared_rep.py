# shared_rep.py
"""
Task Unit과 개별 Task 모두에 대해 ENV/ACT/DES 기반 공유 잠재표현을 생성/저장합니다.
- 최초 실행 시: Task Unit 데이터로 모델 학습 → 모델(.pkl) 저장
- 이후 실행 시: 기존 모델을 로드하고 재학습 없이 공유표현만 계산
- 산출물:
  - multiview_clustering_model.pkl
  - shared_z_task_unit.npy, shared_z_task.npy
  - shared_meta.json (샘플 수/차원 등)
"""
import os
import json
import time
import pickle
import numpy as np
import torch
from multiview_clustering import MultiViewClusteringPipeline

MODEL_PATH = "multiview_clustering_model.pkl"
TU_VEC_PATH = "task_unit_vectors.pkl"
TASK_VEC_PATH = "task_vectors.pkl"
TU_SHARED_PATH = "shared_z_task_unit.npy"
TASK_SHARED_PATH = "shared_z_task.npy"
META_PATH = "shared_meta.json"


def _to_tensors_from_vectors(vectors, device):
    x_env = torch.FloatTensor([v['env_embedding'] for v in vectors]).to(device)
    x_act = torch.FloatTensor([v['act_embedding'] for v in vectors]).to(device)
    x_des = torch.FloatTensor([v['des_embedding'] for v in vectors]).to(device)
    return x_env, x_act, x_des


def _extract_shared_with_model(pipeline: MultiViewClusteringPipeline, vectors):
    device = pipeline.device
    x_env, x_act, x_des = _to_tensors_from_vectors(vectors, device)
    pipeline.model.eval()
    outs = []
    with torch.no_grad():
        bs = 128
        for i in range(0, x_env.shape[0], bs):
            z, _ = pipeline.model.get_shared_representation(
                x_env[i:i+bs], x_act[i:i+bs], x_des[i:i+bs]
            )
            outs.append(z.cpu().numpy())
    return np.vstack(outs)


def generate_shared_rep(
    task_unit_vectors_path: str = "task_unit_vectors_legacy.pkl",  # 🔥 legacy 버전 사용
    task_vectors_path: str = "task_vectors_legacy.pkl",           # 🔥 legacy 버전 사용
    model_save_path: str = MODEL_PATH,
    tu_shared_out: str = TU_SHARED_PATH,
    task_shared_out: str = TASK_SHARED_PATH,
    meta_out: str = META_PATH,
    epochs: int = 30,
    batch_size: int = 16,
):
    with open(task_unit_vectors_path, 'rb') as f:
        task_unit_vectors = pickle.load(f)
    with open(task_vectors_path, 'rb') as f:
        task_vectors = pickle.load(f)

    embedding_dim = len(task_unit_vectors[0]['env_embedding']) if task_unit_vectors else 3072

    pipeline = MultiViewClusteringPipeline(embedding_dim=embedding_dim)

    if os.path.exists(model_save_path):
        pipeline.load_model(model_save_path)
    else:
        data_dict = pipeline.prepare_data_from_vectors(task_unit_vectors, augment=True)
        pipeline.train_model(data_dict, epochs=epochs, batch_size=batch_size)
        pipeline.save_model(model_save_path)

    shared_tu = _extract_shared_with_model(pipeline, task_unit_vectors)
    shared_task = _extract_shared_with_model(pipeline, task_vectors)

    np.save(tu_shared_out, shared_tu)
    np.save(task_shared_out, shared_task)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_save_path,
        "tu_shared_path": tu_shared_out,
        "task_shared_path": task_shared_out,
        "embedding_dim": embedding_dim,
        "latent_dim": pipeline.latent_dim,
        "n_task_units": int(shared_tu.shape[0]),
        "n_tasks": int(shared_task.shape[0])
    }
    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ 공유표현 저장:", tu_shared_out, task_shared_out)
    return shared_tu, shared_task, meta


if __name__ == "__main__":
    generate_shared_rep()