# -*- coding: utf-8 -*-
"""
session_multiview_clustering.py

- OpenAI 임베딩(text-embedding-3-large) 사용
- openaikey.env 로딩
- 멀티뷰 입력: (env, keyword, des)  ← 과거 (env, act, des)도 하위호환 지원
- 데이터 증강(augmentation) 없음
- MiniCOMPLETER 모델 + miniCOMPLETER loss(Contrastive + λ·Dual-Prediction + μ·Reconstruction)
- HDBSCAN 등 클러스터링은 외부 파일에서 호출하도록 별도 제공하지 않음(이 파일은 학습/공유표현 추출 전용)

사용 예시:
    pipeline = SessionMultiViewClusteringPipeline(embedding_dim=3072, latent_dim=256)
    data = pipeline.prepare_data_from_vectors(items)  # items: pkl/jsonl에서 읽은 dict 리스트
    pipeline.train_model(data, epochs=30, batch_size=16, lr=1e-3)
    shared_z = pipeline.extract_shared_representations(data)  # (N, latent_dim)
    pipeline.save_model("session_multiview_model.pkl")

로드:
    pipeline = SessionMultiViewClusteringPipeline().load_model("session_multiview_model.pkl")
    data = pipeline.prepare_data_from_vectors(items)
    shared_z = pipeline.extract_shared_representations(data)
"""

import os
import time
import pickle
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# OpenAI Embeddings 설정
# -----------------------------
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("./openaikey.env")
_client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"   # 고정


def _embed_single(text: str) -> List[float]:
    """단일 텍스트 임베딩 (text-embedding-3-large)"""
    if text is None or str(text).strip() == "":
        text = "unknown"
    resp = _client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def _embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """배치 임베딩 생성. 기본은 외부 생성 벡터 재사용, 누락분만 생성."""
    embs: List[List[float]] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = [t if (t and str(t).strip()) else "unknown" for t in texts[i:i + batch_size]]
        resp = _client.embeddings.create(model=EMBED_MODEL, input=batch)
        embs.extend([d.embedding for d in resp.data])
    return embs


# -----------------------------
# Ontology (coarse ENV)
# -----------------------------
class ENVOntology:
    def __init__(self):
        self.coarse_to_fine = {
            "Browser": ["web/Chrome", "web/Firefox", "web/Edge", "web/Safari", "web/YouTube",
                        "web/Gmail", "web/Chrome-ChatGPT", "web/Chrome-GoogleDocs",
                        "web/Chrome-GoogleCalendar"],
            "File_Manager": ["local/FileExplorer", "local/Finder"],
            "Text_Editor": ["app/Notepad", "app/VSCode", "local/Notepad"],
            "Office": ["app/MSWord", "app/MSExcel"],
            "Communication": ["app/Slack", "app/Discord", "app/Teams"],
            "OS_System": ["local/Windows", "local/Desktop", "local/macOS"],
        }
        self.fine_to_coarse = {}
        for c, fines in self.coarse_to_fine.items():
            for f in fines:
                self.fine_to_coarse[f.lower()] = c

        self.aliases = {
            "chrome": "web/Chrome",
            "google docs": "web/Chrome-GoogleDocs",
            "notepad": "app/Notepad",
            "file explorer": "local/FileExplorer",
            "desktop": "local/Desktop",
            "chatgpt": "web/Chrome-ChatGPT",
            "gmail": "web/Gmail",
            "youtube": "web/YouTube",
        }

    def get_coarse(self, fine_env: str) -> str:
        if not fine_env:
            return "Unknown"
        key = fine_env.strip().lower()
        if key in self.aliases:
            key = self.aliases[key].lower()
        if key in self.fine_to_coarse:
            return self.fine_to_coarse[key]
        # 부분 매칭
        for fine_key, coarse in self.fine_to_coarse.items():
            if key in fine_key or fine_key in key:
                return coarse
        return "Unknown"


# -----------------------------
# 모델
# -----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DualPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z):
        return self.net(z)


class MiniCOMPLETER(nn.Module):
    """
    세 개의 view(env/keyword/des)에 대해
    (1) 인코딩(Projection)
    (2) 상호 예측(DualPredictor)
    (3) 복원(Decoder: linear)
    를 포함하는 소형 COMPLETER.
    """
    def __init__(self, embedding_dim: int = 3072, latent_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        # encoders
        self.f_env = ProjectionHead(embedding_dim, 512, latent_dim)
        self.f_kwd = ProjectionHead(embedding_dim, 512, latent_dim)
        self.f_des = ProjectionHead(embedding_dim, 512, latent_dim)

        # cross-view predictors (z_a -> z_b)
        self.G_env_to_kwd = DualPredictor(latent_dim, 128)
        self.G_kwd_to_env = DualPredictor(latent_dim, 128)
        self.G_env_to_des = DualPredictor(latent_dim, 128)
        self.G_des_to_env = DualPredictor(latent_dim, 128)
        self.G_kwd_to_des = DualPredictor(latent_dim, 128)
        self.G_des_to_kwd = DualPredictor(latent_dim, 128)

        # decoders
        self.g_env = nn.Linear(latent_dim, embedding_dim)
        self.g_kwd = nn.Linear(latent_dim, embedding_dim)
        self.g_des = nn.Linear(latent_dim, embedding_dim)

    def forward(self, x_env, x_kwd, x_des):
        # encode
        z_env = self.f_env(x_env)
        z_kwd = self.f_kwd(x_kwd)
        z_des = self.f_des(x_des)

        # cross-view predict
        z_env_from_kwd = self.G_kwd_to_env(z_kwd)
        z_kwd_from_env = self.G_env_to_kwd(z_env)
        z_env_from_des = self.G_des_to_env(z_des)
        z_des_from_env = self.G_env_to_des(z_env)
        z_kwd_from_des = self.G_des_to_kwd(z_des)
        z_des_from_kwd = self.G_kwd_to_des(z_kwd)

        # reconstruct
        x_env_rec = self.g_env(z_env)
        x_kwd_rec = self.g_kwd(z_kwd)
        x_des_rec = self.g_des(z_des)

        return {
            "z_env": z_env,
            "z_kwd": z_kwd,
            "z_des": z_des,
            "z_env_from_kwd": z_env_from_kwd,
            "z_kwd_from_env": z_kwd_from_env,
            "z_env_from_des": z_env_from_des,
            "z_des_from_env": z_des_from_env,
            "z_kwd_from_des": z_kwd_from_des,
            "z_des_from_kwd": z_des_from_kwd,
            "x_env_rec": x_env_rec,
            "x_kwd_rec": x_kwd_rec,
            "x_des_rec": x_des_rec,
        }

    def get_shared_representation(self, x_env, x_kwd, x_des):
        """
        간단한 가중 평균(shared representation).
        필요 시 가중치 조정(예: coarse ENV 신뢰 ↑).
        """
        out = self.forward(x_env, x_kwd, x_des)
        z_shared = 0.4 * out["z_env"] + 0.3 * out["z_kwd"] + 0.3 * out["z_des"]
        return z_shared, out


# -----------------------------
# Loss (miniCOMPLETER)
# -----------------------------
class MultiViewLoss:
    """
    total = Contrastive + lambda_pre * Dual-Prediction + lambda_rec * Reconstruction

    - Contrastive: InfoNCE(z_env, z_kwd/des, z_kwd, z_des) 평균
    - Dual-Prediction: 각 view 쌍 간 예측 z_hat ≈ z (MSE)
    - Reconstruction: 각 view 복원 x_hat ≈ x (MSE)
    """
    def __init__(self, lambda_pre=0.1, lambda_rec=0.05, temperature=0.1):
        self.lambda_pre = float(lambda_pre)
        self.lambda_rec = float(lambda_rec)
        self.temperature = float(temperature)

    def _info_nce(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.matmul(z1, z2.T) / self.temperature
        exp_sim = torch.exp(sim)
        pos = torch.diag(exp_sim)
        denom = exp_sim.sum(dim=1)
        loss = -torch.log((pos + 1e-8) / (denom + 1e-8))
        return loss.mean()

    def dual_prediction_loss(self, out):
        loss = 0.0
        loss += F.mse_loss(out["z_env_from_kwd"], out["z_env"])
        loss += F.mse_loss(out["z_kwd_from_env"], out["z_kwd"])
        loss += F.mse_loss(out["z_env_from_des"], out["z_env"])
        loss += F.mse_loss(out["z_des_from_env"], out["z_des"])
        loss += F.mse_loss(out["z_kwd_from_des"], out["z_kwd"])
        loss += F.mse_loss(out["z_des_from_kwd"], out["z_des"])
        return loss / 6.0

    def reconstruction_loss(self, out, x_env, x_kwd, x_des):
        loss = 0.0
        loss += F.mse_loss(out["x_env_rec"], x_env)
        loss += F.mse_loss(out["x_kwd_rec"], x_kwd)
        loss += F.mse_loss(out["x_des_rec"], x_des)
        return loss / 3.0

    def total(self, out, x_env, x_kwd, x_des):
        cl = (
            self._info_nce(out["z_env"], out["z_kwd"])
            + self._info_nce(out["z_env"], out["z_des"])
            + self._info_nce(out["z_kwd"], out["z_des"])
        ) / 3.0
        pre = self.dual_prediction_loss(out)
        rec = self.reconstruction_loss(out, x_env, x_kwd, x_des)
        total = cl + self.lambda_pre * pre + self.lambda_rec * rec
        return total, {"contrastive": cl, "prediction": pre, "reconstruction": rec}


# -----------------------------
# Data Processor (env, keyword, des)
# -----------------------------
class SessionDataProcessor:
    """
    세션 단위 멀티뷰 입력 표준화:
    - 기존 키: env_text, act_text, des_text
    - 신규 키(jsonl): env(list[str]), keywords(list[str]), description(str)
    내부 표준: env_text, keyword_text, des_text (+ 각 임베딩)
    """
    def __init__(self):
        self.onto = ENVOntology()

    @staticmethod
    def _to_text_list(val) -> str:
        if isinstance(val, list):
            return ", ".join(map(str, val))
        return str(val) if val is not None else "unknown"

    def normalize_item(self, item: Dict) -> Dict:
        out = dict(item)

        # env_text
        if "env_text" in out:
            env_text = out["env_text"]
        elif "env" in out:
            env_text = self._to_text_list(out["env"])
        else:
            env_text = "unknown"
        out["env_text"] = env_text

        # keyword_text (act_text 대체)
        if "keyword_text" in out:
            kw_text = out["keyword_text"]
        elif "act_text" in out:
            kw_text = out["act_text"]
        elif "keywords" in out:
            kw_text = self._to_text_list(out["keywords"])
        else:
            kw_text = "unknown"
        out["keyword_text"] = kw_text

        # des_text
        if "des_text" in out:
            des_text = out["des_text"]
        elif "description" in out:
            des_text = out["description"]
        elif "desc_text" in out:
            des_text = out["desc_text"]
        else:
            des_text = "unknown"
        out["des_text"] = des_text

        # 임베딩 필드 매핑 (kwd 우선순위: keyword_embedding > act_embedding)
        if "keyword_embedding" in out:
            out["kwd_embedding"] = out["keyword_embedding"]
        elif "act_embedding" in out:
            out["kwd_embedding"] = out["act_embedding"]

        return out

    def prepare_arrays(self, items: List[Dict], embedding_dim_fallback: int = 3072) -> Dict:
        normalized = [self.normalize_item(x) for x in items]

        env_texts = [x["env_text"] for x in normalized]
        kwd_texts = [x["keyword_text"] for x in normalized]
        des_texts = [x["des_text"] for x in normalized]

        # 임베딩 확보 (가능하면 기존 필드 재사용)
        def collect_or_embed(key_texts, emb_key_candidates: List[str]):
            vecs = []
            to_embed_idx = []
            for i, x in enumerate(normalized):
                got = None
                for k in emb_key_candidates:
                    if k in x and isinstance(x[k], (list, np.ndarray)):
                        got = x[k]
                        break
                if got is None:
                    to_embed_idx.append(i)
                    vecs.append(None)
                else:
                    vecs.append(got)

            if to_embed_idx:
                texts = [key_texts[i] for i in to_embed_idx]
                embs = _embed_batch(texts)
                for j, idx in enumerate(to_embed_idx):
                    vecs[idx] = embs[j]

            return np.array(vecs, dtype=np.float32)

        env_emb = collect_or_embed(env_texts, ["env_embedding"])
        kwd_emb = collect_or_embed(kwd_texts, ["kwd_embedding", "keyword_embedding", "act_embedding"])
        des_emb = collect_or_embed(des_texts, ["des_embedding"])

        coarse_envs = [self.onto.get_coarse(t if isinstance(t, str) else str(t)) for t in env_texts]

        # 텐서
        x_env = torch.FloatTensor(env_emb)
        x_kwd = torch.FloatTensor(kwd_emb)
        x_des = torch.FloatTensor(des_emb)

        return {
            "x_env": x_env,
            "x_kwd": x_kwd,
            "x_des": x_des,
            "env_texts": env_texts,
            "keyword_texts": kwd_texts,
            "des_texts": des_texts,
            "coarse_envs": coarse_envs,
            "original_items": normalized,
            "embedding_dim": x_env.shape[1] if x_env.ndim == 2 else embedding_dim_fallback,
        }


# -----------------------------
# Pipeline (학습/추출 전용)
# -----------------------------
class SessionMultiViewClusteringPipeline:
    def __init__(self, embedding_dim=3072, latent_dim=256):
        self.embedding_dim = int(embedding_dim)
        self.latent_dim = int(latent_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[MiniCOMPLETER] = None
        self.processor = SessionDataProcessor()
        print(f"🔧 Device: {self.device}")

    def prepare_data_from_vectors(self, items: List[Dict]) -> Dict:
        print("📊 세션 멀티뷰 입력 준비...")
        data = self.processor.prepare_arrays(items, embedding_dim_fallback=self.embedding_dim)
        # 디바이스로 이동
        data["x_env"] = data["x_env"].to(self.device)
        data["x_kwd"] = data["x_kwd"].to(self.device)
        data["x_des"] = data["x_des"].to(self.device)
        # 실제 embedding_dim 동기화
        self.embedding_dim = int(data["x_env"].shape[1])
        return data

    def train_model(
        self,
        data: Dict,
        epochs: int = 30,
        batch_size: int = 16,
        lr: float = 1e-3,
        lambda_pre: float = 0.1,
        lambda_rec: float = 0.05,
        temperature: float = 0.1,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        cosine_anneal: bool = True,
    ):
        print("🚀 MiniCOMPLETER 학습 시작...")
        self.model = MiniCOMPLETER(self.embedding_dim, self.latent_dim).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if cosine_anneal:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        loss_fn = MultiViewLoss(lambda_pre=lambda_pre, lambda_rec=lambda_rec, temperature=temperature)

        ds = TensorDataset(data["x_env"], data["x_kwd"], data["x_des"])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        for ep in range(epochs):
            self.model.train()
            total = cl = pre = rec = 0.0
            n = 0
            for b_env, b_kwd, b_des in dl:
                opt.zero_grad()
                out = self.model(b_env, b_kwd, b_des)
                loss, parts = loss_fn.total(out, b_env, b_kwd, b_des)
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                opt.step()

                total += float(loss.item())
                cl += float(parts["contrastive"].item())
                pre += float(parts["prediction"].item())
                rec += float(parts["reconstruction"].item())
                n += 1

            if cosine_anneal:
                sched.step()

            if ep == 0 or (ep + 1) % 10 == 0:
                print(f"Epoch {ep+1:02d}/{epochs} | total {total/n:.4f} | CL {cl/n:.4f} | PRE {pre/n:.4f} | REC {rec/n:.4f}")

        print("✅ 학습 완료!")
        return self.model

    def extract_shared_representations(self, data: Dict) -> np.ndarray:
        if self.model is None:
            raise ValueError("모델이 없습니다. 먼저 train_model() 또는 load_model()을 호출하세요.")
        self.model.eval()
        outs = []
        with torch.no_grad():
            bs = 128
            N = data["x_env"].shape[0]
            for i in range(0, N, bs):
                z, _ = self.model.get_shared_representation(
                    data["x_env"][i:i+bs], data["x_kwd"][i:i+bs], data["x_des"][i:i+bs]
                )
                outs.append(z.cpu().numpy())
        shared_z = np.vstack(outs)
        print(f"🧠 공유표현(z_shared): {shared_z.shape}")
        return shared_z

    # -------------------------
    # 저장/로드
    # -------------------------
    def save_model(self, path: str = "session_multiview_model.pkl"):
        payload = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "embedding_dim": self.embedding_dim,
            "latent_dim": self.latent_dim,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"💾 모델 저장: {path}")

    def load_model(self, path: str = "session_multiview_model.pkl"):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.embedding_dim = int(payload.get("embedding_dim", self.embedding_dim))
        self.latent_dim = int(payload.get("latent_dim", self.latent_dim))
        self.model = MiniCOMPLETER(self.embedding_dim, self.latent_dim).to(self.device)
        state = payload.get("model_state_dict")
        if state:
            self.model.load_state_dict(state)
            self.model.eval()
        print(f"📂 모델 로드 완료 (저장 시각: {payload.get('timestamp','?')})")
        return self