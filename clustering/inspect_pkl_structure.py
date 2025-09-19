# -*- coding: utf-8 -*-
"""
inspect_pkl_structure.py

PKL 파일의 구조와 내부 요소를 상세히 분석하는 스크립트입니다.
- 파일 존재 여부 및 크기 확인
- 데이터 타입 및 구조 분석
- 샘플 데이터 출력
- 필드별 통계 정보
- 임베딩 벡터 차원 및 분포 확인

실행 예시:
  python inspect_pkl_structure.py --all
  python inspect_pkl_structure.py --task-units
  python inspect_pkl_structure.py --tasks
  python inspect_pkl_structure.py --sample 3
"""

import os
import pickle
import argparse
import numpy as np
import json
from typing import List, Dict, Any, Union
from collections import Counter


# 파일 경로 설정
TU_VEC_PATH = "task_unit_vectors.pkl"
TASK_VEC_PATH = "task_vectors.pkl"
TU_VEC_BACKUP = "task_unit_vectors_backup.pkl"
TASK_VEC_BACKUP = "task_vectors_backup.pkl"


def _print_separator(title: str, char: str = "=", width: int = 80):
    """구분선 출력"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def _print_subsection(title: str, char: str = "-", width: int = 60):
    """소제목 출력"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def _format_bytes(size_bytes: int) -> str:
    """바이트를 읽기 쉬운 형태로 변환"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def _analyze_vector_field(vectors: List[Dict], field_name: str) -> Dict[str, Any]:
    """벡터 필드의 통계 분석"""
    if not vectors or field_name not in vectors[0]:
        return {"exists": False}
    
    # 첫 번째 벡터로 차원 확인
    first_vec = vectors[0][field_name]
    if not isinstance(first_vec, (list, np.ndarray)):
        return {"exists": False, "error": "Not a vector"}
    
    dimension = len(first_vec)
    
    # 모든 벡터를 numpy 배열로 변환
    try:
        all_vectors = np.array([v[field_name] for v in vectors])
        
        stats = {
            "exists": True,
            "dimension": dimension,
            "count": len(vectors),
            "mean": float(np.mean(all_vectors)),
            "std": float(np.std(all_vectors)),
            "min": float(np.min(all_vectors)),
            "max": float(np.max(all_vectors)),
            "shape": all_vectors.shape
        }
        
        # 첫 번째 벡터의 일부 값들
        stats["sample_values"] = first_vec[:10] if len(first_vec) >= 10 else first_vec
        
        return stats
    except Exception as e:
        return {"exists": True, "error": str(e)}


def _analyze_text_field(vectors: List[Dict], field_name: str) -> Dict[str, Any]:
    """텍스트 필드의 통계 분석"""
    if not vectors or field_name not in vectors[0]:
        return {"exists": False}
    
    values = [v.get(field_name, "") for v in vectors]
    non_empty = [v for v in values if v and str(v).strip()]
    
    # 길이 통계
    lengths = [len(str(v)) for v in values]
    
    # 가장 흔한 값들
    counter = Counter(values)
    most_common = counter.most_common(5)
    
    return {
        "exists": True,
        "total_count": len(values),
        "non_empty_count": len(non_empty),
        "empty_count": len(values) - len(non_empty),
        "avg_length": np.mean(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "most_common": most_common,
        "sample_values": values[:5]
    }


def _analyze_numeric_field(vectors: List[Dict], field_name: str) -> Dict[str, Any]:
    """숫자 필드의 통계 분석"""
    if not vectors or field_name not in vectors[0]:
        return {"exists": False}
    
    values = []
    for v in vectors:
        val = v.get(field_name)
        if isinstance(val, (int, float)):
            values.append(val)
        elif isinstance(val, str) and val.isdigit():
            values.append(int(val))
    
    if not values:
        return {"exists": True, "error": "No numeric values found"}
    
    return {
        "exists": True,
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": min(values),
        "max": max(values),
        "unique_count": len(set(values)),
        "sample_values": values[:10]
    }


def check_file_existence():
    """파일 존재 여부 확인"""
    _print_separator("파일 존재 여부 확인")
    
    files_to_check = [
        (TU_VEC_PATH, "Task Unit 벡터 (메인)"),
        (TASK_VEC_PATH, "Task 벡터 (메인)"),
        (TU_VEC_BACKUP, "Task Unit 벡터 (백업)"),
        (TASK_VEC_BACKUP, "Task 벡터 (백업)")
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {description}")
            print(f"   경로: {file_path}")
            print(f"   크기: {_format_bytes(size)}")
        else:
            print(f"❌ {description}")
            print(f"   경로: {file_path} (파일 없음)")
        print()


def load_and_inspect_pkl(file_path: str, file_type: str):
    """PKL 파일 로드 및 구조 분석"""
    if not os.path.exists(file_path):
        print(f"❌ {file_path} 파일이 존재하지 않습니다.")
        return None
    
    _print_separator(f"{file_type} 구조 분석 - {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ 파일 로드 성공")
        print(f"📊 데이터 타입: {type(data)}")
        print(f"📊 레코드 수: {len(data) if isinstance(data, (list, tuple)) else 'N/A'}")
        
        if isinstance(data, list) and len(data) > 0:
            return analyze_record_structure(data, file_type)
        else:
            print("⚠️ 빈 리스트이거나 예상과 다른 데이터 구조입니다.")
            return data
            
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None


def analyze_record_structure(data: List[Dict], file_type: str):
    """레코드 구조 상세 분석"""
    if not data:
        return data
    
    first_record = data[0]
    
    _print_subsection("기본 정보")
    print(f"총 레코드 수: {len(data)}")
    print(f"첫 번째 레코드 타입: {type(first_record)}")
    
    if isinstance(first_record, dict):
        print(f"필드 수: {len(first_record.keys())}")
        print(f"필드 목록: {list(first_record.keys())}")
    
    _print_subsection("필드별 상세 분석")
    
    if isinstance(first_record, dict):
        # 각 필드 타입별로 분석
        for field_name, field_value in first_record.items():
            print(f"\n🔍 필드: '{field_name}'")
            print(f"   타입: {type(field_value)}")
            
            if isinstance(field_value, (list, np.ndarray)) and len(field_value) > 0:
                # 벡터 필드 분석
                if isinstance(field_value[0], (int, float)):
                    stats = _analyze_vector_field(data, field_name)
                    if stats.get("exists"):
                        print(f"   벡터 차원: {stats.get('dimension', 'N/A')}")
                        print(f"   평균: {stats.get('mean', 0):.6f}")
                        print(f"   표준편차: {stats.get('std', 0):.6f}")
                        print(f"   범위: [{stats.get('min', 0):.6f}, {stats.get('max', 0):.6f}]")
                        print(f"   샘플 값: {stats.get('sample_values', [])}")
                else:
                    print(f"   리스트 길이: {len(field_value)}")
                    print(f"   샘플 값: {field_value[:3] if len(field_value) >= 3 else field_value}")
            
            elif isinstance(field_value, str):
                # 텍스트 필드 분석
                stats = _analyze_text_field(data, field_name)
                if stats.get("exists"):
                    print(f"   비어있지 않은 값: {stats.get('non_empty_count')}/{stats.get('total_count')}")
                    print(f"   평균 길이: {stats.get('avg_length', 0):.1f}")
                    print(f"   길이 범위: [{stats.get('min_length', 0)}, {stats.get('max_length', 0)}]")
                    if stats.get('most_common'):
                        print(f"   가장 흔한 값: {stats['most_common'][0] if stats['most_common'] else 'N/A'}")
            
            elif isinstance(field_value, (int, float)):
                # 숫자 필드 분석
                stats = _analyze_numeric_field(data, field_name)
                if stats.get("exists") and not stats.get("error"):
                    print(f"   평균: {stats.get('mean', 0):.2f}")
                    print(f"   범위: [{stats.get('min', 0)}, {stats.get('max', 0)}]")
                    print(f"   고유 값 수: {stats.get('unique_count', 0)}")
            
            else:
                print(f"   값: {str(field_value)[:100]}{'...' if len(str(field_value)) > 100 else ''}")
    
    return data


def show_sample_records(data: List[Dict], file_type: str, num_samples: int = 2):
    """샘플 레코드 출력"""
    if not data:
        return
    
    _print_separator(f"{file_type} 샘플 레코드")
    
    num_samples = min(num_samples, len(data))
    
    for i in range(num_samples):
        print(f"\n📋 샘플 #{i+1}:")
        record = data[i]
        
        if isinstance(record, dict):
            for key, value in record.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 10:
                    # 긴 벡터는 차원과 일부 값만 표시
                    print(f"  {key}: [{len(value)}차원 벡터] {value[:5]}...{value[-2:]}")
                elif isinstance(value, str) and len(value) > 100:
                    # 긴 텍스트는 일부만 표시
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {record}")
        
        if i < num_samples - 1:
            print("-" * 50)


def compare_embeddings(data: List[Dict], file_type: str):
    """임베딩 벡터들 비교 분석"""
    if not data:
        return
    
    _print_separator(f"{file_type} 임베딩 비교 분석")
    
    # 임베딩 필드 찾기
    embedding_fields = []
    first_record = data[0]
    
    for field_name, field_value in first_record.items():
        if 'embedding' in field_name or field_name == 'z_shared':
            if isinstance(field_value, (list, np.ndarray)):
                embedding_fields.append(field_name)
    
    if not embedding_fields:
        print("임베딩 필드를 찾을 수 없습니다.")
        return
    
    print(f"발견된 임베딩 필드: {embedding_fields}")
    
    for field in embedding_fields:
        print(f"\n🧠 {field} 분석:")
        stats = _analyze_vector_field(data, field)
        
        if stats.get("exists") and not stats.get("error"):
            print(f"   차원: {stats['dimension']}")
            print(f"   샘플 수: {stats['count']}")
            print(f"   값 범위: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"   평균: {stats['mean']:.6f}")
            print(f"   표준편차: {stats['std']:.6f}")
            
            # 벡터 간 유사도 체크 (첫 5개)
            if stats['count'] >= 2:
                vectors = np.array([v[field] for v in data[:5]])
                similarities = []
                for i in range(len(vectors)):
                    for j in range(i+1, len(vectors)):
                        # 코사인 유사도 계산
                        cos_sim = np.dot(vectors[i], vectors[j]) / (
                            np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
                        )
                        similarities.append(cos_sim)
                
                if similarities:
                    print(f"   샘플 간 코사인 유사도 평균: {np.mean(similarities):.4f}")
                    print(f"   샘플 간 코사인 유사도 범위: [{min(similarities):.4f}, {max(similarities):.4f}]")


def export_structure_summary(tu_data: List[Dict], task_data: List[Dict]):
    """구조 요약을 JSON으로 내보내기"""
    _print_separator("구조 요약 내보내기")
    
    summary = {
        "timestamp": __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
        "task_units": {
            "count": len(tu_data) if tu_data else 0,
            "fields": list(tu_data[0].keys()) if tu_data else []
        },
        "tasks": {
            "count": len(task_data) if task_data else 0,
            "fields": list(task_data[0].keys()) if task_data else []
        }
    }
    
    # 임베딩 필드 정보 추가
    if tu_data:
        embedding_info = {}
        for field in tu_data[0].keys():
            if 'embedding' in field or field == 'z_shared':
                if isinstance(tu_data[0][field], (list, np.ndarray)):
                    embedding_info[field] = len(tu_data[0][field])
        summary["task_units"]["embeddings"] = embedding_info
    
    if task_data:
        embedding_info = {}
        for field in task_data[0].keys():
            if 'embedding' in field or field == 'z_shared':
                if isinstance(task_data[0][field], (list, np.ndarray)):
                    embedding_info[field] = len(task_data[0][field])
        summary["tasks"]["embeddings"] = embedding_info
    
    # JSON 파일로 저장
    output_file = "pkl_structure_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 구조 요약이 {output_file}에 저장되었습니다.")
    print(f"📄 요약 내용:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='PKL 파일 구조 상세 분석')
    parser.add_argument('--all', action='store_true', help='모든 분석 수행')
    parser.add_argument('--task-units', action='store_true', help='Task Unit 파일만 분석')
    parser.add_argument('--tasks', action='store_true', help='Task 파일만 분석')
    parser.add_argument('--sample', type=int, default=2, help='표시할 샘플 레코드 수')
    parser.add_argument('--export', action='store_true', help='구조 요약 JSON 내보내기')
    parser.add_argument('--files-only', action='store_true', help='파일 존재 여부만 확인')
    
    args = parser.parse_args()
    
    if args.files_only:
        check_file_existence()
        return
    
    tu_data = None
    task_data = None
    
    if args.all or args.task_units or (not args.tasks and not args.all):
        tu_data = load_and_inspect_pkl(TU_VEC_PATH, "Task Unit")
        if tu_data:
            show_sample_records(tu_data, "Task Unit", args.sample)
            compare_embeddings(tu_data, "Task Unit")
    
    if args.all or args.tasks or (not args.task_units and not args.all):
        task_data = load_and_inspect_pkl(TASK_VEC_PATH, "Task")
        if task_data:
            show_sample_records(task_data, "Task", args.sample)
            compare_embeddings(task_data, "Task")
    
    if args.export and (tu_data or task_data):
        export_structure_summary(tu_data, task_data)
    
    # 파일 존재 확인은 항상 마지막에
    if args.all:
        check_file_existence()


if __name__ == '__main__':
    main()