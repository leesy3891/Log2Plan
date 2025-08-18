import openai
import os
import glob
import re
import pickle
import time
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from dotenv import load_dotenv

# 캐시 시스템 import
from long_term_plan_cache import LongTermPlanCache, bayesian_avg

load_dotenv('./openaikey.env')
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key:
    print("API 키가 성공적으로 설정되었습니다.")

from openai import OpenAI
client = openai.OpenAI()

# ==================== 유틸리티 함수 ====================

def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def embed(text):
    """단일 텍스트 임베딩"""
    return client.embeddings.create(
        model="text-embedding-3-large", 
        input=[text]
    ).data[0].embedding

def create_embeddings(texts):
    """기존 호환성을 위한 배치 텍스트 임베딩 (단일 임베딩)"""
    batch_size = 20
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])
    
    return all_embeddings

def create_embeddings_separated(texts_env, texts_act, texts_des):
    """
    ENV, ACT, DES를 개별적으로 임베딩 생성
    """
    batch_size = 20
    
    def embed_batch(texts):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings
    
    print("ENV 태그 임베딩 생성 중...")
    env_embeddings = embed_batch(texts_env)
    
    print("ACT 태그 임베딩 생성 중...")
    act_embeddings = embed_batch(texts_act)
    
    print("Description 임베딩 생성 중...")
    des_embeddings = embed_batch(texts_des)
    
    return env_embeddings, act_embeddings, des_embeddings

def weighted_similarity(item1, item2, env_weight=0.3, act_weight=0.2, des_weight=0.5):
    """
    ENV:ACT:DES = 3:2:5 가중합으로 유사도 계산
    """
    env_sim = cosine_similarity(item1['env_embedding'], item2['env_embedding'])
    act_sim = cosine_similarity(item1['act_embedding'], item2['act_embedding'])
    des_sim = cosine_similarity(item1['des_embedding'], item2['des_embedding'])
    
    return env_weight * env_sim + act_weight * act_sim + des_weight * des_sim

def weighted_query_similarity(query_emb, item, env_weight=0.3, act_weight=0.2, des_weight=0.5):
    """
    쿼리와 아이템 간 가중 유사도 계산
    """
    if isinstance(query_emb, dict):
        # 쿼리도 분리된 임베딩인 경우
        env_sim = cosine_similarity(query_emb['env'], item['env_embedding'])
        act_sim = cosine_similarity(query_emb['act'], item['act_embedding'])
        des_sim = cosine_similarity(query_emb['des'], item['des_embedding'])
    else:
        # 쿼리가 단일 임베딩인 경우 (기존 호환성)
        env_sim = cosine_similarity(query_emb, item['env_embedding'])
        act_sim = cosine_similarity(query_emb, item['act_embedding'])
        des_sim = cosine_similarity(query_emb, item['des_embedding'])
    
    return env_weight * env_sim + act_weight * act_sim + des_weight * des_sim

# ==================== 로그 파싱 ====================

def read_labeled_logs():
    """labeled_logs 디렉토리에서 모든 로그 파일을 읽어옵니다."""
    logs = []
    for file_path in glob.glob('labeled_logs/*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            logs.append({"file_path": file_path, "content": f.read()})
    return logs

def extract_task_unit_headers(gpt_output: str) -> str:
    """GPT 출력에서 Task Unit 헤더만 추출"""
    pattern = r'Task\s*Unit\s*:?\s*(\*\*.*?\*\*)'
    headers = re.findall(pattern, gpt_output)
    if not headers:
        print("⚠ No Task Unit headers matched.")
    return "\n".join([f"Task Unit : {h}" for h in headers])

def extract_tasks(unit_text: str):
    """Task Unit에서 개별 Task# 블록들을 추출"""
    pattern = r'(Task#\d+:[\s\S]*?)(?=\nTask#\d+:|$)'
    return [m.group(1).strip() for m in re.finditer(pattern, unit_text)]

def extract_task_units(logs):
    """로그에서 Task Unit 추출 및 파싱"""
    task_units = []
    
    for log in logs:
        content = log['content']
        file_name = os.path.basename(log['file_path'])
        
        # Task Unit 패턴 매칭
        pattern = r'Task Unit #(\d+): \*\*(.*?)\*\*\s+Description: (.*?)(?=Task#\d+:|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for idx, (unit_num, name, description) in enumerate(matches, start=1):
            unit_id = f"{file_name}:Unit{unit_num}"
            unit_start = content.find(f"Task Unit #{unit_num}")
            next_header = f"Task Unit #{int(unit_num)+1}"
            unit_end = content.find(next_header) if next_header in content else len(content)
            unit_content = content[unit_start:unit_end].strip()
            
            env_tags, act_tag, title = parse_tags(name)
            
            task_units.append({
                "id": unit_id,
                "file_path": log['file_path'],
                "name": name.strip(),
                "description": description.strip(),
                "content": unit_content,
                "env_tag": env_tags,
                "act_tag": act_tag,
                "title": title
            })
    
    return task_units

def parse_tags(name):
    """ENV[], ACT[], Title 파싱"""
    env_match = re.search(r'ENV\[(.*?)\]', name)
    act_match = re.search(r'ACT\[(.*?)\]', name)
    
    env_raw = env_match.group(1).strip() if env_match else ''
    env_tags = [e.strip() for e in env_raw.split(',')] if env_raw else []
    
    act_tag = act_match.group(1).strip() if act_match else ''
    
    # Title = ACT] 뒤의 부분
    title_part = re.sub(r'.*ACT\[[^\]]*\]\s*', '', name).strip()
    title = re.sub(r'\s*\(.*?\)\s*$', '', title_part).strip()
    
    return env_tags, act_tag, title

# ==================== DBSCAN 벡터 DB ====================

def create_vector_db(eps_factor=1.5, min_samples=3, auto_adjust=True):
    """
    개선된 DBSCAN 클러스터링을 사용한 벡터 DB 생성
    ENV, ACT, DES를 개별 임베딩으로 처리
    """
    db_file = "task_unit_vectors_dbscan_separated.pkl"
    
    if os.path.exists(db_file):
        print("기존 벡터 DB를 로드합니다...")
        with open(db_file, "rb") as f:
            return pickle.load(f)
    
    print("새로운 벡터 DB를 생성합니다...")
    
    # 1. 로그 읽기 및 Task Unit 추출
    logs = read_labeled_logs()
    print(f"{len(logs)}개의 로그 파일을 읽었습니다.")
    
    task_units = extract_task_units(logs)
    print(f"{len(task_units)}개의 작업 단위를 추출했습니다.")
    
    # 2. ENV, ACT, DES 텍스트 분리
    texts_env = []
    texts_act = []
    texts_des = []
    
    for unit in task_units:
        # ENV: 환경 태그들을 쉼표로 연결
        env_text = ', '.join(unit['env_tag']) if unit['env_tag'] else 'unknown'
        texts_env.append(env_text)
        
        # ACT: 액션 태그
        act_text = unit['act_tag'] if unit['act_tag'] else 'unknown'
        texts_act.append(act_text)
        
        # DES: 제목 + 설명 결합
        des_text = f"{unit['title']} - {unit['description']}" if unit['title'] and unit['description'] else unit['title'] or unit['description'] or 'unknown'
        texts_des.append(des_text)
    
    # 3. 개별 임베딩 생성
    print("ENV, ACT, DES 개별 임베딩을 생성합니다...")
    env_embeddings, act_embeddings, des_embeddings = create_embeddings_separated(
        texts_env, texts_act, texts_des
    )
    
    # 4. 클러스터링을 위한 결합 임베딩 생성 (가중 평균)
    print("클러스터링용 결합 임베딩을 생성합니다...")
    combined_embeddings = []
    for i in range(len(task_units)):
        # ENV:ACT:DES = 3:2:5 가중 평균으로 결합
        env_emb = np.array(env_embeddings[i])
        act_emb = np.array(act_embeddings[i])
        des_emb = np.array(des_embeddings[i])
        
        combined = 0.3 * env_emb + 0.2 * act_emb + 0.5 * des_emb
        combined_embeddings.append(combined.tolist())
    
    # 5. DBSCAN 클러스터링 (결합 임베딩 사용)
    print("DBSCAN 클러스터링을 수행합니다...")
    
    # 개선된 eps 계산 방식
    dists = pairwise_distances(combined_embeddings, metric="cosine")
    tri = dists[np.triu_indices_from(dists, k=1)]
    
    # 기본 통계
    median, std = np.median(tri), np.std(tri)
    percentile_25 = np.percentile(tri, 25)
    percentile_75 = np.percentile(tri, 75)
    
    # 더 관대한 eps 계산 (여러 방법 중 최대값 선택)
    eps_methods = [
        median + eps_factor * std,           # 기존 방식 (더 큰 계수)
        percentile_25 + 0.3 * (percentile_75 - percentile_25),  # IQR 기반
        np.mean(tri) + 0.8 * std             # 평균 기반
    ]
    eps = float(max(eps_methods))  # 가장 관대한 값 선택
    
    print(f"거리 통계 - 중앙값: {median:.4f}, 표준편차: {std:.4f}")
    print(f"DBSCAN 파라미터: eps={eps:.4f}, min_samples={min_samples}")
    
    # 자동 조정 모드
    best_clustering = None
    best_params = None
    
    if auto_adjust:
        print("최적 파라미터를 탐색합니다...")
        
        # 여러 파라미터 조합 시도
        param_combinations = [
            (eps, min_samples),
            (eps * 1.2, min_samples),
            (eps * 1.5, min_samples),
            (eps, max(2, min_samples - 1)),
            (eps * 1.3, max(2, min_samples - 1))
        ]
        
        for test_eps, test_min_samples in param_combinations:
            dbscan = DBSCAN(eps=test_eps, min_samples=test_min_samples, metric='cosine')
            test_labels = dbscan.fit_predict(combined_embeddings)
            
            n_clusters = len(set(test_labels)) - (1 if -1 in test_labels else 0)
            n_noise = list(test_labels).count(-1)
            noise_ratio = n_noise / len(test_labels)
            
            # 좋은 클러스터링 조건: 
            # 1) 3개 이상의 클러스터
            # 2) 노이즈 비율 70% 미만
            # 3) 클러스터당 평균 아이템 수가 적절함
            if n_clusters >= 3 and noise_ratio < 0.7:
                avg_cluster_size = (len(test_labels) - n_noise) / max(n_clusters, 1)
                if avg_cluster_size >= 2:  # 클러스터당 최소 2개 아이템
                    best_clustering = test_labels
                    best_params = (test_eps, test_min_samples)
                    print(f"✅ 적절한 파라미터 발견: eps={test_eps:.4f}, min_samples={test_min_samples}")
                    print(f"   클러스터 수: {n_clusters}, 노이즈 비율: {noise_ratio:.1%}")
                    break
            
            print(f"   시도: eps={test_eps:.4f}, min_samples={test_min_samples} → 클러스터:{n_clusters}, 노이즈:{noise_ratio:.1%}")
    
    # 최적 파라미터로 최종 클러스터링
    if best_clustering is not None:
        cluster_labels = best_clustering
        eps, min_samples = best_params
    else:
        print("⚠️ 자동 조정 실패. 기본 파라미터 사용")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = dbscan.fit_predict(combined_embeddings)
    
    # 최종 클러스터 정보
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"📊 최종 클러스터링 결과:")
    print(f"   클러스터 수: {n_clusters}")
    print(f"   노이즈 포인트: {n_noise} ({n_noise/len(cluster_labels):.1%})")
    
    # 클러스터별 상세 정보
    if n_clusters > 0:
        cluster_sizes = {}
        for label in cluster_labels:
            if label != -1:  # 노이즈 제외
                cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        print(f"   클러스터 크기 분포: {dict(sorted(cluster_sizes.items()))}")
    
    # 6. 벡터 DB 구성 (개별 임베딩 + 결합 임베딩 모두 저장)
    vector_db = []
    for i, unit in enumerate(task_units):
        vector_db.append({
            "id": unit["id"],
            "file_path": unit["file_path"],
            "content": unit["content"],
            "env_tag": unit["env_tag"],
            "act_tag": unit["act_tag"],
            "title": unit["title"],
            "description": unit["description"],
            "env_embedding": env_embeddings[i],      # ENV 개별 임베딩
            "act_embedding": act_embeddings[i],      # ACT 개별 임베딩
            "des_embedding": des_embeddings[i],      # DES 개별 임베딩
            "embedding": combined_embeddings[i],     # 결합 임베딩 (기존 호환성)
            "cluster": int(cluster_labels[i])        # DBSCAN 클러스터 ID
        })
    
    # 7. 저장
    with open(db_file, "wb") as f:
        pickle.dump(vector_db, f)
    
    print(f"벡터 데이터베이스가 {db_file}에 저장되었습니다.")
    return vector_db

# ==================== 다양성 기반 검색 ====================

def search_hier_mmr(query_emb, vector_db, top_clusters=5, k_per=3, lambda_div=0.4, fallback_to_similarity=True):
    """
    개선된 클러스터 기반 MMR 검색 (가중 유사도 사용)
    """
    # 클러스터별 대표 임베딩 계산 (노이즈 제외)
    reps = defaultdict(list)
    noise_items = []  # 노이즈 아이템 별도 보관
    
    for v in vector_db:
        if v['cluster'] != -1:  # 정상 클러스터
            reps[v['cluster']].append(v['embedding'])
        else:  # 노이즈 아이템
            noise_items.append(v)
    
    # 각 클러스터의 중심 임베딩
    rep_embs = {cid: np.mean(rep, axis=0) for cid, rep in reps.items() if rep}
    
    print(f"🔍 활용 가능한 클러스터: {len(rep_embs)}개, 노이즈 아이템: {len(noise_items)}개")
    
    # 클러스터가 너무 적으면 유사도 기반 검색으로 대체
    if len(rep_embs) < 2 and fallback_to_similarity:
        print("⚠️ 클러스터 수 부족 → 가중 유사도 기반 검색으로 대체")
        return search_by_weighted_similarity(query_emb, vector_db, top_k=top_clusters * k_per)
    
    # 클러스터 순위 매기기 (가중 유사도 사용)
    cid_rank = []
    for cid in rep_embs:
        # 클러스터 대표와 쿼리 간 가중 유사도
        if isinstance(query_emb, dict):
            # 쿼리가 분리된 임베딩인 경우
            sim = cosine_similarity(query_emb.get('combined', query_emb.get('embedding')), rep_embs[cid])
        else:
            # 쿼리가 단일 임베딩인 경우
            sim = cosine_similarity(query_emb, rep_embs[cid])
        cid_rank.append((sim, cid))
    
    cid_rank.sort(key=lambda x: x[0], reverse=True)
    cid_rank = [cid for _, cid in cid_rank[:top_clusters]]
    
    picked = []
    
    # 각 클러스터에서 MMR 선택 (가중 유사도 사용)
    for cid in cid_rank:
        cand = [v for v in vector_db if v['cluster'] == cid]
        
        # 가중 유사도로 점수 계산
        scores = []
        for v in cand:
            if isinstance(query_emb, dict):
                sim = weighted_query_similarity(query_emb, v)
            else:
                # 기존 호환성을 위해 결합 임베딩 사용
                sim = cosine_similarity(query_emb, v['embedding'])
            scores.append((v, sim))
        
        # MMR 선택
        selected = []
        while scores and len(selected) < k_per:
            mmr = []
            for v, s in scores:
                # 가중 유사도로 다양성 계산
                max_sim = 0
                for x in selected:
                    div_sim = weighted_similarity(v, x)
                    max_sim = max(max_sim, div_sim)
                
                mmr_score = s - lambda_div * max_sim
                mmr.append((mmr_score, v))
            
            v_best = max(mmr, key=lambda t: t[0])[1]
            selected.append(v_best)
            scores = [(v, s) for v, s in scores if v is not v_best]
        
        picked.extend(selected)
    
    # 결과가 부족하면 노이즈 아이템에서 보충
    if len(picked) < top_clusters * k_per // 2 and noise_items:
        print(f"🔄 결과 부족 → 노이즈 아이템 {len(noise_items)}개에서 보충")
        
        noise_scores = []
        for v in noise_items:
            if isinstance(query_emb, dict):
                sim = weighted_query_similarity(query_emb, v)
            else:
                sim = cosine_similarity(query_emb, v['embedding'])
            noise_scores.append((v, sim))
        
        noise_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 기존 선택과 중복되지 않는 노이즈 아이템 추가
        needed = (top_clusters * k_per) - len(picked)
        for v, score in noise_scores[:needed * 2]:
            if v not in picked:
                picked.append(v)
                if len(picked) >= top_clusters * k_per:
                    break
    
    print(f"✅ 최종 선택된 아이템: {len(picked)}개")
    return picked

def search_by_weighted_similarity(query_emb, vector_db, top_k=15):
    """
    가중 유사도 기반 검색 (폴백 옵션)
    """
    print("📊 가중 유사도 기반 검색을 수행합니다...")
    
    # 모든 아이템에 대해 가중 유사도 계산
    scored = []
    for v in vector_db:
        if isinstance(query_emb, dict):
            sim = weighted_query_similarity(query_emb, v)
        else:
            sim = cosine_similarity(query_emb, v['embedding'])
        scored.append((sim, v))
    
    # 유사도순 정렬
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # 다양성을 위한 간단한 필터링 (가중 유사도 사용)
    selected = []
    threshold = 0.95  # 너무 유사한 것들 제거
    
    for sim, v in scored:
        # 이미 선택된 것과 너무 유사하지 않은지 확인
        too_similar = False
        for selected_v in selected:
            if weighted_similarity(v, selected_v) > threshold:
                too_similar = True
                break
        
        if not too_similar:
            selected.append(v)
            if len(selected) >= top_k:
                break
    
    print(f"✅ 가중 유사도 기반 선택: {len(selected)}개")
    return selected
    
    for v in vector_db:
        if v['cluster'] != -1:  # 정상 클러스터
            reps[v['cluster']].append(v['embedding'])
        else:  # 노이즈 아이템
            noise_items.append(v)
    
    # 각 클러스터의 중심 임베딩
    rep_embs = {cid: np.mean(rep, axis=0) for cid, rep in reps.items() if rep}
    
    print(f"🔍 활용 가능한 클러스터: {len(rep_embs)}개, 노이즈 아이템: {len(noise_items)}개")
    
    # 클러스터가 너무 적으면 유사도 기반 검색으로 대체
    if len(rep_embs) < 2 and fallback_to_similarity:
        print("⚠️ 클러스터 수 부족 → 유사도 기반 검색으로 대체")
        return search_by_similarity_only(query_emb, vector_db, top_k=top_clusters * k_per)
    
    # 클러스터 순위 매기기
    cid_rank = sorted(
        rep_embs, 
        key=lambda c: cosine_similarity(query_emb, rep_embs[c]), 
        reverse=True
    )[:top_clusters]
    
    picked = []
    
    # 각 클러스터에서 MMR 선택
    for cid in cid_rank:
        cand = [v for v in vector_db if v['cluster'] == cid]
        scores = [(v, cosine_similarity(query_emb, v['embedding'])) for v in cand]
        
        # MMR 선택
        selected = []
        while scores and len(selected) < k_per:
            mmr = []
            for v, s in scores:
                max_sim = max([
                    cosine_similarity(v['embedding'], x['embedding']) 
                    for x in selected
                ] or [0])
                
                mmr_score = s - lambda_div * max_sim
                mmr.append((mmr_score, v))
            
            v_best = max(mmr, key=lambda t: t[0])[1]
            selected.append(v_best)
            scores = [(v, s) for v, s in scores if v is not v_best]
        
        picked.extend(selected)
    
    # 결과가 부족하면 노이즈 아이템에서 보충
    if len(picked) < top_clusters * k_per // 2 and noise_items:
        print(f"🔄 결과 부족 → 노이즈 아이템 {len(noise_items)}개에서 보충")
        
        noise_scores = [
            (v, cosine_similarity(query_emb, v['embedding'])) 
            for v in noise_items
        ]
        noise_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 기존 선택과 중복되지 않는 노이즈 아이템 추가
        needed = (top_clusters * k_per) - len(picked)
        for v, score in noise_scores[:needed * 2]:  # 여유있게 후보 확보
            if v not in picked:
                picked.append(v)
                if len(picked) >= top_clusters * k_per:
                    break
    
    print(f"✅ 최종 선택된 아이템: {len(picked)}개")
    return picked

def search_by_similarity_only(query_emb, vector_db, top_k=15):
    """
    클러스터링 없이 순수 유사도 기반 검색 (폴백 옵션)
    """
    print("📊 순수 유사도 기반 검색을 수행합니다...")
    
    # 모든 아이템에 대해 유사도 계산
    scored = []
    for v in vector_db:
        sim = cosine_similarity(query_emb, v['embedding'])
        scored.append((sim, v))
    
    # 유사도순 정렬
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # 다양성을 위한 간단한 필터링
    selected = []
    threshold = 0.95  # 너무 유사한 것들 제거
    
    for sim, v in scored:
        # 이미 선택된 것과 너무 유사하지 않은지 확인
        too_similar = False
        for selected_v in selected:
            if cosine_similarity(v['embedding'], selected_v['embedding']) > threshold:
                too_similar = True
                break
        
        if not too_similar:
            selected.append(v)
            if len(selected) >= top_k:
                break
    
    print(f"✅ 유사도 기반 선택: {len(selected)}개")
    return selected

def search_diverse_similar_tasks(query: str, vector_db: list, top_k: int = 9):
    """
    단계적 다이버시티 선택 (가중 유사도 사용)
    """
    query = query.strip()
    if not query:
        raise ValueError("❌ Provided query is empty. Cannot create embedding.")
    
    # 쿼리를 ENV, ACT, DES로 분리하여 임베딩 생성
    print("🔍 쿼리 분석 및 분리된 임베딩 생성...")
    
    # 간단한 쿼리 파싱 (실제로는 더 정교할 수 있음)
    env_text = "unknown"  # 기본값
    act_text = "unknown"  # 기본값
    des_text = query      # 전체 쿼리를 설명으로 사용
    
    # Task Unit 헤더에서 ENV, ACT 추출 시도
    env_match = re.search(r'ENV\[(.*?)\]', query)
    act_match = re.search(r'ACT\[(.*?)\]', query)
    
    if env_match:
        env_text = env_match.group(1)
    if act_match:
        act_text = act_match.group(1)
        # ACT 이후 부분을 설명으로 사용
        des_text = re.sub(r'.*ACT\[[^\]]*\]\s*', '', query).strip()
    
    # 개별 임베딩 생성
    query_env_emb = embed(env_text)
    query_act_emb = embed(act_text)
    query_des_emb = embed(des_text)
    
    query_emb_dict = {
        'env': query_env_emb,
        'act': query_act_emb,
        'des': query_des_emb,
        'combined': 0.3 * np.array(query_env_emb) + 0.2 * np.array(query_act_emb) + 0.5 * np.array(query_des_emb)
    }
    
    # 가중 유사도 계산 및 정렬
    scored = []
    for idx, item in enumerate(vector_db):
        sim = weighted_query_similarity(query_emb_dict, item)
        scored.append((idx, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    selected, excluded = [], set()
    
    def exclude_most_similar(to_idx):
        """가장 유사한 항목 제외 (가중 유사도 사용)"""
        base_item = vector_db[to_idx]
        best_j, best_sim = None, -1
        for j, _ in scored:
            if j in excluded or j == to_idx:
                continue
            sim = weighted_similarity(base_item, vector_db[j])
            if sim > best_sim:
                best_sim, best_j = sim, j
        if best_j is not None:
            excluded.add(best_j)
    
    stage_targets = [3, 3, 3]  # 3-3-3
    for stage_size in stage_targets:
        for idx, _ in scored:
            if idx in excluded:
                continue
            selected.append(idx)
            excluded.add(idx)
            exclude_most_similar(idx)
            if len(selected) % stage_size == 0:
                break
    
    selected = selected[:top_k]
    return [vector_db[i] for i in selected]

def top_similar_tasks_per_current(content_text: str, candidate_units: list, top_n: int = 2):
    """
    현재 Task#별로 가장 유사한 후보 Task# 선택 (가중 유사도 사용)
    """
    cur_tasks = extract_tasks(content_text)
    if not cur_tasks:
        return []
    
    # 후보 Task# 전부 수집
    cand_tasks = []
    for unit in candidate_units:
        for t in extract_tasks(unit['content']):
            cand_tasks.append(t)
    
    # 현재 Task들을 ENV, ACT, DES로 분리하여 임베딩
    print("🔍 현재 Task들의 분리된 임베딩 생성...")
    cur_env_texts, cur_act_texts, cur_des_texts = [], [], []
    
    for task in cur_tasks:
        # Task에서 ENV, ACT 추출
        env_match = re.search(r'ENV\[(.*?)\]', task)
        act_match = re.search(r'ACT\[(.*?)\]', task)
        desc_match = re.search(r'Description:\s*(.*?)(?=\n|$)', task)
        
        env_text = env_match.group(1) if env_match else "unknown"
        act_text = act_match.group(1) if act_match else "unknown"
        desc_text = desc_match.group(1) if desc_match else task[:100]  # 첫 100자를 설명으로
        
        cur_env_texts.append(env_text)
        cur_act_texts.append(act_text)
        cur_des_texts.append(desc_text)
    
    # 후보 Task들도 마찬가지로 분리
    cand_env_texts, cand_act_texts, cand_des_texts = [], [], []
    
    for task in cand_tasks:
        env_match = re.search(r'ENV\[(.*?)\]', task)
        act_match = re.search(r'ACT\[(.*?)\]', task)
        desc_match = re.search(r'Description:\s*(.*?)(?=\n|$)', task)
        
        env_text = env_match.group(1) if env_match else "unknown"
        act_text = act_match.group(1) if act_match else "unknown"
        desc_text = desc_match.group(1) if desc_match else task[:100]
        
        cand_env_texts.append(env_text)
        cand_act_texts.append(act_text)
        cand_des_texts.append(desc_text)
    
    # 모든 텍스트 임베딩 생성
    all_env_texts = cur_env_texts + cand_env_texts
    all_act_texts = cur_act_texts + cand_act_texts
    all_des_texts = cur_des_texts + cand_des_texts
    
    env_embs, act_embs, des_embs = create_embeddings_separated(all_env_texts, all_act_texts, all_des_texts)
    
    # 분리
    cur_env_embs = env_embs[:len(cur_tasks)]
    cur_act_embs = act_embs[:len(cur_tasks)]
    cur_des_embs = des_embs[:len(cur_tasks)]
    
    cand_env_embs = env_embs[len(cur_tasks):]
    cand_act_embs = act_embs[len(cur_tasks):]
    cand_des_embs = des_embs[len(cur_tasks):]
    
    # Task별 top-n 선정 (가중 유사도 사용)
    selected = []
    for i in range(len(cur_tasks)):
        sims = []
        for j in range(len(cand_tasks)):
            # 가중 유사도 계산
            env_sim = cosine_similarity(cur_env_embs[i], cand_env_embs[j])
            act_sim = cosine_similarity(cur_act_embs[i], cand_act_embs[j])
            des_sim = cosine_similarity(cur_des_embs[i], cand_des_embs[j])
            
            weighted_sim = 0.3 * env_sim + 0.2 * act_sim + 0.5 * des_sim
            sims.append((weighted_sim, cand_tasks[j]))
        
        sims.sort(key=lambda x: x[0], reverse=True)
        top_examples = [t for _, t in sims[:top_n]]
        selected.extend(top_examples)
    
    return selected

# ==================== Coherence 점수 (기존 로직) ====================

ACT_TRANSITION = defaultdict(int)
TRANS_PROB = {}

def build_act_transition(vector_db):
    """ACT 전이 확률 계산"""
    for v in vector_db:
        acts = []
        for t in extract_tasks(v['content']):
            match = re.search(r'ACT\[(.*?)\]', t)
            if match:
                acts.append(match.group(1))
        
        for a, b in zip(acts, acts[1:]):
            ACT_TRANSITION[(a, b)] += 1
    
    total = sum(ACT_TRANSITION.values())
    for k, v in ACT_TRANSITION.items():
        TRANS_PROB[k] = v / total

def coherence(path):
    """경로의 coherence 점수 계산"""
    acts = []
    for t in path:
        match = re.search(r'ACT\[(.*?)\]', t)
        if match:
            acts.append(match.group(1))
    
    if len(acts) < 2:
        return 0
    
    return np.mean([
        TRANS_PROB.get((acts[i], acts[i+1]), 0) 
        for i in range(len(acts)-1)
    ])

def fill_missing(path, query_emb, vector_db):
    """누락된 ACT 작업 보강"""
    seen = set()
    for p in path:
        match = re.search(r'ACT\[(.*?)\]', p)
        if match:
            seen.add(match.group(1))
    
    for v in vector_db:
        if v['act_tag'] in seen:
            continue
        if cosine_similarity(query_emb, v['embedding']) > 0.8:
            path.append(v['content'])
            seen.add(v['act_tag'])
    
    return path

# ==================== GPT 명령어 처리 ====================

def process_command(command):
    """GPT API를 사용하여 명령어를 구조화된 Task Unit으로 변환"""
    prompt = f"""Your job is to break down a command given by the user, which can be executed in user's Desktop, Windows. 
    Identify tasks by breaking down the command, and label them. Task Unit containes a sequence of Tasks to execute the command.

    Create a structured labeling in the following format:
    
    Task Unit : **ENV[environment/platform-subdomains] ACT[action_category/specific_action] Descriptive Title**  

    Task#1: ENV[environment/platform-subdomains] ACT[action]  
    Description: [Brief description of this specific subtask, max 20 words]

    Task#2: ENV[environment/platform-subdomains] ACT[action] 
    Description: [Brief description of this specific subtask, max 20 words]

    [List of numbered tasks goes here, keeping original numbering]

    Always use "Task Unit : **ENV[] ACT[] Title** format because I need to parse your response.
    Follow these rules for the labeling:

    1. ENV tag format:
       - local/[program] - For local desktop applications (e.g., local/FileExplorer)
       - web/[sitename] - For web browsing (e.g., web/Chrome-University Portal, web/Chrome-Shopping Platform)
       - app/[appname] - For specific applications (e.g., app/Excel, app/VSCode)
       
    2. ACT tag format:
    - [action1, action2,..]
    Examples:
    | ACT Tag |
    |------------|
    | ACT[search, filter_sort] |
    | ACT[add_to_cart, product_purchase] |
    | ACT[open_document] |

    3. For Task Units, make a sequence of environments which are fit for fulfilling user's command, if multiple envs needs to be involved:
       - ENV[web/Chrome-a Web to get a summary] when a user requests a summary of a paper
       - ENV[local/FileExplorer,app/Excel] for a unit involving File Explorer and Excel

    4. Use flexible, semantically meaningful ACT tags based on behavior, not only from a fixed list.

    User Command:
    {command}"""

    try:
        print("GPT API 호출 중...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing and breaking down commands into task."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI API 호출 오류: {e}")
        return None

# ==================== 결과 토큰화 ====================

def tokenizer(output):
    """LLM 출력을 구조화된 형태로 파싱"""
    parsed_list = []
    for line in output.strip().split("\n"):
        match = re.match(r"(\d+)\.\s*(\d+)#([^,]+),\s*(.+)", line)
        if match:
            assist_bit = match.group(2).strip()
            action = match.group(3).strip()
            obj = match.group(4).strip()
            parsed_list.append([assist_bit, action, obj])
        else:
            print(f"Warning: Unrecognized task format - {line}")
    
    return parsed_list

# ==================== 로그 기록 ====================

def log_result(query, similar_units, output, llm_output, log_dir="result_logs"):
    """생성 과정과 결과를 로그 파일로 저장"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"result_{timestamp}.txt")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Query: {query}\n\n")
        
        # 유사한 Task Unit 정보
        f.write(f"Similar Task Units ({len(similar_units)}):\n")
        for i, unit in enumerate(similar_units):
            cluster_info = f"Cluster[{unit.get('cluster', 'N/A')}]"
            f.write(f"{i+1}. {unit['id']} - {cluster_info} ENV[{unit['env_tag']}] ACT[{unit['act_tag']}] Title[{unit.get('title', '')}]\n")
        f.write("\n")
        
        # LLM 출력
        f.write("LLM Output:\n")
        f.write(llm_output)
        f.write("\n\n")
        
        # 최종 파싱된 결과
        f.write("Final Parsed Output:\n")
        for i, step in enumerate(output):
            f.write(f"{i+1}. {step[0]}#{step[1]}, {step[2]}\n")
    
    print(f"결과가 {log_file}에 저장되었습니다.")
    return log_file

# ==================== 사용자 피드백 ====================

def get_user_feedback(plan_id: str) -> int:
    """1-5점 피드백 수집"""
    print("\n📝 플랜 품질을 평가해주세요:")
    print("1점: 매우 나쁨 (전혀 작동하지 않음)")
    print("2점: 나쁨 (많은 오류 있음)")
    print("3점: 보통 (일부 작동함)")
    print("4점: 좋음 (대부분 잘 작동함)")
    print("5점: 매우 좋음 (완벽하게 작동함)")
    
    while True:
        try:
            score = int(input("점수를 입력하세요 (1-5): "))
            if 1 <= score <= 5:
                cache = LongTermPlanCache()
                cache.add_feedback(plan_id, score)
                print(f"✅ {score}점 피드백이 저장되었습니다!")
                cache.close()
                return score
            else:
                print("1-5 사이의 숫자를 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")

# ==================== 메인 GlobalPlanner ====================

def globalPlanner(command, content, use_cache=True, sim_thresh=0.75, quality_cut=4.0, top_k_cache=5):
    """
    통합된 GlobalPlanner - DBSCAN 클러스터링 + 캐시 시스템
    """
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    
    # 캐시 시스템 초기화
    cache = LongTermPlanCache()
    plan_id = None
    negative_examples = []
    
    try:
        # 명령어 임베딩 생성
        command_emb = np.array(embed(command))
        
        # 1. 캐시 조회
        if use_cache:
            cached_plans = cache.get_candidates(
                command, command_emb,
                top_k=top_k_cache,
                sim_thresh=sim_thresh,
                include_archived=False
            )
            
            if cached_plans:
                print(f"{CYAN}🔍 캐시에서 {len(cached_plans)}개의 유사한 플랜을 발견했습니다.{RESET}")
                
                high_q, low_q = [], []
                for p in cached_plans:
                    q = bayesian_avg(p.score_sum, p.score_cnt)
                    (high_q if q >= quality_cut else low_q).append((q, p))
                
                if high_q:  # 고품질 플랜 바로 사용
                    high_q.sort(key=lambda x: x[0], reverse=True)
                    best_quality, best_plan = high_q[0]
                    print(f"{CYAN}📋 고품질 캐시 플랜 사용 (품질={best_quality:.2f}, 사용={best_plan.score_cnt}){RESET}")
                    
                    # 실행 가능한 형태로 변환
                    if isinstance(best_plan.plan_json, dict) and "tasks" in best_plan.plan_json:
                        output = best_plan.plan_json["tasks"]
                    else:
                        output = tokenizer(str(best_plan.plan_json))
                    
                    plan_id = best_plan.plan_id
                    
                    # 캐시된 플랜 내용 표시
                    print(f"\n📋 캐시된 플랜 내용:")
                    for i, task in enumerate(output, 1):
                        print(f"  {i}. {task[0]}#{task[1]}, {task[2]}")
                    
                    # 피드백 요청
                    get_user_feedback(plan_id)
                    return output, plan_id
                
                else:
                    negative_examples = [p for _, p in low_q]
                    print(f"{YELLOW}⚠️ 고품질 없음 → 저품질 {len(negative_examples)}개를 부정 예시로 사용{RESET}")
        
        # 2. 새로운 플랜 생성
        print(f"{CYAN}🔄 새로운 플랜을 생성합니다...{RESET}")
        
        # 2-1. 부정 예시 프롬프트 생성
        negative_prompt = ""
        if negative_examples:
            neg_tasks = []
            for neg_plan in negative_examples[:2]:
                pj = neg_plan.plan_json
                if isinstance(pj, dict):
                    if "tasks" in pj:
                        neg_tasks.extend(pj["tasks"][:5])
                    elif "llm_output" in pj:
                        neg_tasks.extend(tokenizer(pj["llm_output"])[:5])
            
            if neg_tasks:
                neg_text = "\n".join(
                    f"{i+1}. {t[0]}#{t[1]}, {t[2]}" for i, t in enumerate(neg_tasks)
                )
                negative_prompt = (
                    "다음은 과거 낮은 평가(≤3점)를 받은 접근 예시입니다. "
                    "이 패턴을 반복하지 말고 더 나은 계획을 제시하세요.\n"
                    f"{neg_text}\n"
                    "위와 같은 순서/행동/설명을 피하세요."
                )
        
        # 2-2. 벡터 데이터베이스 로드 및 검색 (개선된 파라미터 사용)
        vector_db = create_vector_db(eps_factor=1.5, min_samples=3, auto_adjust=True)
        build_act_transition(vector_db)
        
        print("유사한 작업을 검색합니다...")
        query_text = extract_task_unit_headers(content)
        
        # 쿼리를 ENV, ACT, DES로 분리하여 임베딩 생성
        print("🔍 쿼리 분석 및 분리된 임베딩 생성...")
        
        env_text = "unknown"
        act_text = "unknown" 
        des_text = query_text
        
        # Task Unit 헤더에서 ENV, ACT 추출
        env_match = re.search(r'ENV\[(.*?)\]', query_text)
        act_match = re.search(r'ACT\[(.*?)\]', query_text)
        
        if env_match:
            env_text = env_match.group(1)
        if act_match:
            act_text = act_match.group(1)
            des_text = re.sub(r'.*ACT\[[^\]]*\]\s*', '', query_text).strip()
        
        # 분리된 쿼리 임베딩 생성
        query_env_emb = embed(env_text)
        query_act_emb = embed(act_text)
        query_des_emb = embed(des_text)
        
        query_emb_dict = {
            'env': query_env_emb,
            'act': query_act_emb, 
            'des': query_des_emb,
            'combined': (0.3 * np.array(query_env_emb) + 
                        0.2 * np.array(query_act_emb) + 
                        0.5 * np.array(query_des_emb)).tolist()
        }
        
        print(f"📊 분리된 임베딩 생성 완료 - ENV: '{env_text}', ACT: '{act_text}', DES: '{des_text[:50]}...'")
        
        # MMR 기반 검색 (가중 유사도 사용)
        similar_units = search_hier_mmr(query_emb_dict, vector_db, top_clusters=5, k_per=3)
        print(f"{len(similar_units)}개의 유사한 작업을 찾았습니다.")
        
        # Task별 예시 선택
        examples_list = top_similar_tasks_per_current(content, similar_units, top_n=2)
        
        # Coherence 기반 정렬
        examples_list = sorted(examples_list, key=lambda seq: coherence([seq]), reverse=True)
        
        # 누락된 ACT 보강 (결합 임베딩 사용)
        filled_examples = fill_missing(examples_list, query_emb_dict['combined'], vector_db)
        
        examples_text = "\n\n".join(filled_examples)
        print(f"{CYAN}Examples to show to the prompt: {examples_text}{RESET}")
        
        # 2-3. LLM 시스템 메시지 구성
        sys_msgs = [
            {"role": "system", "content": "you are a Desktop automation agent GUI control automation task. your OS is Windows 10. make a list of subtasks which is a combination of the given Events to fulfill the given task."},
            {"role": "system", "content": "Each subtask is in 'assist bit# Event, objects' form. User assist bit is a bit to tell if a Event should be done by the user or target for the task was not given. It should normally be 0, but should be 1 when user action is needed. ex: 'input user ID and PW', 'drag the area you want to copy', 'doubleclick the version that fits'. object is the target of the action."},
            {"role": "system", "content": "Given Events(#18): click/rightclick/drag/scroll/press/text input/open/close/switch focus/go-to/save/copy/paste/delete/rename/login/repeat/wait.  ex: 0#switch focus, back to Excel sheet, 0#click, Image button"},
            {"role": "system", "content": "As an exception, when the Event is repeat, the subtask should be in a form of 'assist bit# repeat, task#a-b, on object_A' like '9. 0#repeat, 3-8, on fileB, 10. 0#repeat, 3-8, on fileC...'. Event: switch focus is when focused window needs change(not needed when focus is naturally changed by the previous Event, like opening Chrome). Event: go-to means navigation to another website inside current window.(not needed when navigation happens by the previous Event, like clicking a hyperlink)"},
            {"role": "system", "content": "if subtask is 'press (blank)', (blank) should be KEYBOARD_KEYS in PyautoGUI."},
            {"role": "system", "content": "If possible, choose using shortcut keys on Windows over click/rightclick/doubleclick. ex:use 'f2' instead of 'click 'Rename', 'ctrl+shift+n' to create new folder, 'ctrl+g' to group selected objects.' Use Chrome as your browser on Web tasks, and open files/apps directely using without traversing through file explorer. (EX: 1. 0#press, win\n2. 0#text input, Filename.txt\n 3. press, enter)"},
            {"role": "system", "content": "IMPORTANT: Avoid unnecessary use of 'drag'. Drag events often occur meaninglessly in user logs, so do not insert 'drag' unless it clearly reflects a user action like selecting or moving specific items."},
            {"role": "system", "content": "Here is one example, Example: execute kakaotalk -> (1. 0#open, kakaotalk\n2. 1#login, on kakaotalk)"},
            {"role": "system", "content": "You are an automation planner AI. You should take inspiration from past task approach. Do NOT just copy and be creative."},
            {"role": "system", "content": "You are given multiple past Task Units. Each Task Unit contains tasks. You must **select the most relevant individual tasks**, and combine them to create a new plan"},
            {"role": "system", "content": "Choose the best matching task(s) from different Task Units."},
            {"role": "system", "content": "Follow these steps: 1. Analyze the user request and infer needed ENV and ACT. 2. From the given examples, extract useful task(s) (e.g., Task#1 from Task Unit A, Task#2 from Task Unit B, etc.). 3. Reorder and adjust tasks to match the goal logically and efficiently."},
            {"role": "system", "content": "IMPORTANT: Format your output EXACTLY as follows, with no additional text, headings, or explanations:\n1. 0#press, win\n2. 0#text input, chrome\n3. 0#press, enter\n...\n\nDo not include any other text or explanation - ONLY the numbered list with the exact format shown above."}
        ]
        
        if negative_prompt:
            sys_msgs.insert(1, {"role": "system", "content": negative_prompt})
        
        user_msgs = [
            {"role": "user", "content": f"Here are some related past Task for inspiration (these are just examples, not exact solutions):\n{examples_text}"},
            {"role": "user", "content": f"Task: {command}"}
        ]
        
        messages = sys_msgs + user_msgs
        
        # 2-4. LLM 호출
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[m for m in messages if m],
            max_tokens=4000,
            temperature=0.4
        )
        
        print(f"Given command: {content}")
        llm_output = response.choices[0].message.content
        print(f"{MAGENTA}Task Splitted: {llm_output}{RESET}")
        
        output = tokenizer(llm_output)
        
        # 2-5. 새로운 플랜을 캐시에 저장
        retrieved_log_ids = [task.get('id', 'unknown') for task in similar_units]
        plan_json = {"tasks": output, "llm_output": llm_output}
        
        plan_id = cache.insert_plan(
            command_text=command,
            command_emb=command_emb,
            retrieved_logs=retrieved_log_ids,
            plan_json=plan_json
        )
        
        print(f"{CYAN}💾 새로운 플랜이 캐시에 저장되었습니다. ID: {plan_id[:8]}...{RESET}")
        
        # 3. 사용자 피드백
        get_user_feedback(plan_id)
        
        # 4. 결과 로그 저장
        log_result(content, similar_units, output, llm_output)
        
        return output, plan_id
    
    finally:
        cache.close()

# ==================== 메인 실행 부분 ====================

def main():
    """메인 실행 함수"""
    print("=== 통합된 GlobalPlanner (DBSCAN + 캐시) ===")
    
    # 사용자 명령어 입력
    command = input("Command: ")
    print(f"Given command: {command}")
    
    # GPT로 명령어 처리
    content = process_command(command)
    if not content:
        print("❌ 명령어 처리에 실패했습니다.")
        return
    
    print(f"Processed Command: {content}")
    
    # GlobalPlanner 실행
    try:
        task_list, plan_id = globalPlanner(command, content)
        
        print(f"\n🎯 최종 생성된 작업 수: {len(task_list)}")
        print("\n📋 최종 작업 계획:")
        for i, task in enumerate(task_list, 1):
            print(f"  {i}. {task[0]}#{task[1]}, {task[2]}")
        
        print(f"\n✅ 플랜 ID: {plan_id[:8]}...")
        
    except Exception as e:
        print(f"❌ GlobalPlanner 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()