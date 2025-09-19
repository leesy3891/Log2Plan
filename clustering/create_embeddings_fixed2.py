# -*- coding: utf-8 -*-
"""
create_embeddings_fixed_v2.py — 기존 출력(PKL, 메타데이터 등)은 그대로 유지하면서
요청한 포맷의 JSON(Task / Task Unit)도 추가로 생성하는 스크립트.

v2 변경점 (중요 버그픽스)
- Task Unit의 "Description"이 뒤쪽 제어과정/Title까지 끌려오는 문제 수정
  -> unit 블록에서 Description만 깔끔히 잘라내는 헬퍼(_extract_unit_description) 추가
  -> 정제된 description을 desc_text와 JSON 저장에 사용

✔ 기대 동작
- 기존 create_embeddings.py와 동일하게 임베딩 생성 및 다음 파일 유지/생성:
  - task_unit_vectors.pkl, task_vectors.pkl, embeddings_metadata.json 등
- 추가 출력(JSON)
  - ./steps_json/task_units_fixed.jsonl   (Task Unit 포맷)
  - ./steps_json/tasks_fixed.jsonl        (Task 포맷)

실행 예시:
  python create_embeddings_fixed_v2.py --run
  python create_embeddings_fixed_v2.py --check

필수 디렉토리/파일:
  - labeled_logs/*.txt
  - openaikey.env (OPENAI_API_KEY 포함)
"""

import os
import re
import glob
import json
import time
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv
load_dotenv('./openaikey.env')

try:
    from openai import OpenAI
    _client = OpenAI()
except Exception:
    _client = None  # 오프라인 확인 모드에서도 동작하도록 허용

# =========================
# 공통 유틸
# =========================

def _read_logs() -> List[Dict[str, Any]]:
    logs = []
    for p in glob.glob('labeled_logs/*.txt'):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                logs.append({
                    'file_path': p,
                    'file_name': os.path.basename(p),
                    'content': f.read()
                })
        except Exception as e:
            print(f"⚠️ 파일 읽기 실패: {p} - {e}")
    if not logs:
        print("⚠️ labeled_logs 디렉토리에 .txt 로그가 없습니다.")
    else:
        print(f"📂 로그 파일 {len(logs)}개 읽음")
    return logs

_env_pat = re.compile(r'ENV\[(.*?)\]')
_act_pat = re.compile(r'ACT\[(.*?)\]')


def _parse_tags(name: str) -> Tuple[List[str], str, str]:
    env_match = _env_pat.search(name)
    act_match = _act_pat.search(name)

    env_raw = env_match.group(1).strip() if env_match else ''
    env_tags = [e.strip() for e in env_raw.split(',')] if env_raw else []
    act_tag = act_match.group(1).strip() if act_match else ''

    # Title = 마지막 "]" 뒤의 본문 (괄호의 (0~29) 같은 범위 표현은 제거)
    title_part = re.sub(r'.*ACT\[[^\]]*\]\s*', '', name).strip()
    title = re.sub(r'\s*\(.*?\)\s*$', '', title_part).strip()
    return env_tags, act_tag, title


def _extract_unit_block(content: str, unit_num: str) -> str:
    m = re.search(rf'Task Unit #({unit_num}):([\s\S]*?)(?=Task Unit #\d+:|$)', content)
    return (m.group(0).strip() if m else '')


def _extract_all_numbered_steps(unit_content: str) -> Dict[int, str]:
    """Unit 전체에서 번호가 있는 모든 step들을 추출.
    저장 시에는 번호 접두부("n.")를 제거하고 0#부터 시작하는 텍스트만 사용.
    """
    steps = {}
    for line in unit_content.splitlines():
        line = line.strip()
        m = re.match(r'^(\d+)\.\s*(0#.+)$', line)
        if m:
            idx = int(m.group(1))
            zero_hash = m.group(2)  # 예: 0#press, win
            steps[idx] = zero_hash
    return steps


def _extract_step_range_from_task(task_content: str) -> Tuple[int, int]:
    m = re.search(r'\((\d+)~(\d+)\)', task_content)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _extract_unit_description(unit_block: str) -> str:
    """Task Unit 블록에서 Description만 깔끔히 추출.
    - "Description:" 이후 텍스트를 처음 빈 줄(\n\n)이나 구획 신호(\nTask#, \nENV[, \nACT[, \nTitle:) 전까지로 제한
    - 뒤섞여 들어오는 Title/제어 과정이 섞이지 않도록 방지
    """
    m = re.search(r'Description:\s*([\s\S]*)', unit_block)
    if not m:
        return ''
    tail = m.group(1)
    cut_candidates = []
    for pat in [r'\n\s*\n', r'\nTask#', r'\nENV\[', r'\nACT\[', r'\nTitle:']:
        mm = re.search(pat, tail)
        if mm:
            cut_candidates.append(mm.start())
    cut_at = min(cut_candidates) if cut_candidates else len(tail)
    desc = tail[:cut_at].strip()
    # 같은 줄 말미에 Title이 붙은 경우 대비
    desc = re.sub(r'Title:.*$', '', desc).strip()
    return desc


def _extract_units_and_tasks(logs: List[Dict[str, Any]]):
    task_units = []
    tasks = []

    # description 캡처는 하되, 실제 저장에는 사용하지 않고 unit_block에서 재계산된 description 사용
    unit_pat = re.compile(r'Task Unit #(\d+): \*\*(.*?)\*\*\s+Description: ([\s\S]*?)(?=Task Unit #\d+:|$)')
    task_pat_inside_unit = re.compile(r'(Task#\d+:[\s\S]*?)(?=\nTask#\d+:|$)')

    for log in logs:
        content = log['content']
        file_name = log['file_name']
        units = unit_pat.findall(content)
        for _, (unit_num, name, _desc_ignored) in enumerate(units, start=1):
            unit_uid = f"TU_{len(task_units):04d}"
            unit_block = _extract_unit_block(content, unit_num)
            all_steps_map = _extract_all_numbered_steps(unit_block)

            # 개별 Task 블록들
            task_blocks = task_pat_inside_unit.findall(unit_block)
            env_tags, act_tag, title = _parse_tags(name)

            # Description 정제 (중요!)
            description = _extract_unit_description(unit_block)

            # Task Unit JSON 레코드 준비 (steps: 탭 연결)
            unit_steps_list = [all_steps_map[k] for k in sorted(all_steps_map.keys())]
            tu_json = {
                "unique_id": unit_uid,
                "file_name": file_name,
                "unit_number": str(unit_num),
                "name": name.strip(),
                "description": description.strip(),
                "task_count": len(task_blocks),
                "env_tag": env_tags,
                "act_tag": act_tag,
                "title": title,
                "env_text": ", ".join(env_tags) if env_tags else "unknown",
                "act_text": act_tag if act_tag else "unknown",
                "desc_text": f"{title} - {description.strip()}" if title and description.strip() else (title or description.strip() or "unknown"),
                "steps": "\t".join(unit_steps_list) if unit_steps_list else "",
            }
            task_units.append(tu_json)

            # Task JSON 레코드들
            for t_idx, task_block in enumerate(task_blocks):
                t_uid = f"T_{len(tasks):05d}"
                env_m = _env_pat.search(task_block)
                act_m = _act_pat.search(task_block)
                desc_m = re.search(r'Description:\s*(.*?)(?=\n|$)', task_block, re.MULTILINE)

                env_text = env_m.group(1) if env_m else "unknown"
                act_text = act_m.group(1) if act_m else "unknown"
                desc_text = (desc_m.group(1).strip() if desc_m else "unknown")

                s, e = _extract_step_range_from_task(task_block)
                step_range = f"({s}~{e})" if s is not None else ""
                chosen_steps = []
                if s is not None and e is not None:
                    for k in range(s, e + 1):
                        if k in all_steps_map:
                            chosen_steps.append(all_steps_map[k])  # 번호 제거된 "0#..."만 탭 연결

                task_json = {
                    "unique_id": t_uid,
                    "parent_unit_id": unit_uid,
                    "parent_unit_name": title,
                    "env_text": env_text,
                    "act_text": act_text,
                    "desc_text": desc_text,
                    "file_name": file_name,
                    "step_range": step_range,
                    "step_count": len(chosen_steps),
                    "steps": "\t".join(chosen_steps) if chosen_steps else "",
                }
                tasks.append(task_json)

    return task_units, tasks

# =========================
# 임베딩 (기존 출력 유지)
# =========================

def _emb_batch(texts: List[str], model: str = "text-embedding-3-large", batch_size: int = 20):
    if _client is None:
        # 테스트/체크 모드: 더미 벡터
        import numpy as np
        dim = 3072 if model == "text-embedding-3-large" else 1536
        return [np.zeros(dim, dtype=float).tolist() for _ in texts]

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = _client.embeddings.create(model=model, input=batch)
            all_vecs.extend([d.embedding for d in resp.data])
        except Exception as e:
            print(f"⚠️ 배치 임베딩 실패 @ {i}: {e}")
            # 개별 보정
            for t in batch:
                try:
                    r = _client.embeddings.create(model=model, input=[t])
                    all_vecs.append(r.data[0].embedding)
                except Exception:
                    import numpy as np
                    dim = 3072 if model == "text-embedding-3-large" else 1536
                    all_vecs.append(np.random.normal(0, 0.1, dim).tolist())
    return all_vecs


def _attach_embeddings_task_units(tus: List[Dict[str, Any]], model: str = "text-embedding-3-large"):
    env = [', '.join(tu['env_tag']) if tu['env_tag'] else 'unknown' for tu in tus]
    act = [tu['act_tag'] if tu['act_tag'] else 'unknown' for tu in tus]
    des = [tu['desc_text'] for tu in tus]

    env_e = _emb_batch(env, model)
    act_e = _emb_batch(act, model)
    des_e = _emb_batch(des, model)

    out = []
    for i, tu in enumerate(tus):
        item = dict(tu)
        item['env_embedding'] = env_e[i]
        item['act_embedding'] = act_e[i]
        item['des_embedding'] = des_e[i]
        out.append(item)
    return out


def _attach_embeddings_tasks(ts: List[Dict[str, Any]], model: str = "text-embedding-3-large"):
    env = [t['env_text'] for t in ts]
    act = [t['act_text'] for t in ts]
    des = [t['desc_text'] for t in ts]

    env_e = _emb_batch(env, model)
    act_e = _emb_batch(act, model)
    des_e = _emb_batch(des, model)

    out = []
    for i, t in enumerate(ts):
        item = dict(t)
        item['env_embedding'] = env_e[i]
        item['act_embedding'] = act_e[i]
        item['des_embedding'] = des_e[i]
        out.append(item)
    return out


# =========================
# 저장 루틴
# =========================

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _save_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"   ✅ {os.path.basename(path)} 저장 (lines={len(records)})")


def _save_pkl(task_units_with_emb: List[Dict[str, Any]], tasks_with_emb: List[Dict[str, Any]]):
    # content/steps는 JSON에 이미 있으므로 PKL은 임베딩 중심 유지 (원 코드 호환)
    def strip_steps(items):
        out = []
        for it in items:
            d = {k: v for k, v in it.items() if k != 'steps'}
            out.append(d)
        return out

    tu_pkl = strip_steps(task_units_with_emb)
    t_pkl = strip_steps(tasks_with_emb)

    with open('task_unit_vectors.pkl', 'wb') as f:
        pickle.dump(tu_pkl, f)
    with open('task_vectors.pkl', 'wb') as f:
        pickle.dump(t_pkl, f)

    # 레거시 파일도 동일하게 유지
    with open('task_unit_vectors_legacy.pkl', 'wb') as f:
        pickle.dump(tu_pkl, f)
    with open('task_vectors_legacy.pkl', 'wb') as f:
        pickle.dump(t_pkl, f)

    print("   ✅ PKL 저장 완료 (기존 파이프라인 호환)")


def _save_metadata(logs_cnt: int, tu_cnt: int, t_cnt: int, dim: int, extra: Dict[str, Any] = None):
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_task_units": tu_cnt,
        "total_tasks": t_cnt,
        "embedding_model": "text-embedding-3-large",
        "embedding_dimension": dim,
        "content_in_json": True,
        "pkl_content_excluded": True,
    }
    if extra:
        meta.update(extra)
    with open('embeddings_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("   ✅ embeddings_metadata.json 저장 완료")


# =========================
# 메인 파이프라인
# =========================

def run_pipeline():
    logs = _read_logs()
    if not logs:
        return False

    print("📊 Task Unit / Task 추출…")
    task_units, tasks = _extract_units_and_tasks(logs)
    print(f"   Task Units: {len(task_units)}  |  Tasks: {len(tasks)}")

    print("🧠 임베딩 생성… (ENV/ACT/DES)")
    tus_with_emb = _attach_embeddings_task_units(task_units)
    ts_with_emb = _attach_embeddings_tasks(tasks)

    print("💾 기존 출력 유지(PKL) 저장…")
    dim = len(tus_with_emb[0]['env_embedding']) if tus_with_emb else 0
    _save_pkl(tus_with_emb, ts_with_emb)

    print("📝 JSON 생성 (요청 포맷)…")
    outdir = './steps_json'
    _ensure_dir(outdir)

    # 요청 포맷에 맞춘 JSONL 저장
    _save_jsonl(os.path.join(outdir, 'task_units_fixed.jsonl'), task_units)
    _save_jsonl(os.path.join(outdir, 'tasks_fixed.jsonl'), tasks)

    _save_metadata(
        logs_cnt=len(logs),
        tu_cnt=len(task_units),
        t_cnt=len(tasks),
        dim=dim,
        extra={
            "json_outputs": {
                "task_units_fixed": os.path.join(outdir, 'task_units_fixed.jsonl'),
                "tasks_fixed": os.path.join(outdir, 'tasks_fixed.jsonl'),
            }
        }
    )

    print("\n✅ 모든 작업 완료!")
    return True


def check_outputs():
    files = [
        'task_unit_vectors.pkl', 'task_vectors.pkl', 'embeddings_metadata.json',
        './steps_json/task_units_fixed.jsonl', './steps_json/tasks_fixed.jsonl'
    ]
    print("📦 생성물 점검:")
    for p in files:
        if os.path.exists(p):
            print(f"  ✅ {p} — {os.path.getsize(p):,} bytes")
        else:
            print(f"  ❌ {p} — 없음")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', help='파이프라인 실행(임베딩+PKL+JSON)')
    ap.add_argument('--check', action='store_true', help='출력물 존재 여부 점검')
    args = ap.parse_args()

    if args.check:
        check_outputs()
    else:
        run_pipeline()