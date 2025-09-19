import os
import openai
import re
import time
import glob
from dotenv import load_dotenv

# 환경 변수에서 API 키 로드
load_dotenv('./openaikey.env')
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print("API 키가 성공적으로 설정되었습니다.")
else:
    print("❌ API 키를 설정해주세요.")

from openai import OpenAI
client = OpenAI()

def get_unlabeled_log_files(log_dir="action_logs", output_dir="labeled_logs"):
    """
    미처리된 변환된 로그 파일 목록을 가져옵니다.
    """
    # 변환된 action_log 파일 목록을 가져옴
    converted_files = glob.glob(os.path.join(log_dir, "converted_action_log_*.txt"))
    
    # 이미 라벨링된 파일들 확인
    labeled_files = glob.glob(os.path.join(output_dir, "labeled_converted_action_log_*.txt"))
    
    # 타임스탬프 추출 정규식
    pattern = r"converted_action_log_(\d+)\.txt"
    
    # 이미 라벨링된 파일의 타임스탬프 추출
    labeled_timestamps = {
        re.search(r"labeled_converted_action_log_(\d+)\.txt", os.path.basename(f)).group(1)
        for f in labeled_files 
        if re.search(r"labeled_converted_action_log_(\d+)\.txt", os.path.basename(f))
    }
    
    # 미처리된 파일 필터링
    unlabeled_logs = [
        f for f in converted_files 
        if re.search(pattern, os.path.basename(f)) and 
        re.search(pattern, os.path.basename(f)).group(1) not in labeled_timestamps
    ]
    
    return unlabeled_logs

def generate_output_filename(converted_log_file, output_dir="labeled_logs"):
    """
    변환된 로그 파일명을 기반으로 라벨링된 파일명 생성
    """
    base_name = os.path.basename(converted_log_file)
    match = re.search(r"converted_action_log_(\d+)\.txt", base_name)
    
    if match:
        timestamp = match.group(1)
        return os.path.join(output_dir, f"labeled_converted_action_log_{timestamp}.txt")
    else:
        return os.path.join(output_dir, f"labeled_{base_name}")

def read_log_file(file_path):
    """
    로그 파일의 내용을 읽어옵니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return None

def process_log_with_gpt(log_content):
    """
    GPT API를 사용하여 로그 내용을 처리합니다.
    """
    # 프롬프트 템플릿
    prompt = f"""These are series of processes collected and preprocessed from user's Desktop, Windows. your job is to group and label the tasks.
Each of these processes are in index. 0#Event: [GUI element name, element control type, (difference in window names: optional)] format.
Ignore redundant keys. ex:cmd+shift_r+Ccmd+shift_rcmd+hcmd+rcmd+ocmd+mcmd...>> Shift_r+Chrom
As an exception, Error message is recorded when click/rightclick/doubleclick occured but failed to extract UI_element name. 
Identify and label Tasks units based on the collected data in this format:

Task Unit #1: **[SHORT DESCRIPTIVE TITLE] (start_index~end_index)**  
Description: [One-sentence overview of what the user is trying to accomplish in this unit, max 50 words]

Task#1: [start_index]~[end_index]  
Description: [Brief description of this specific subtask, max 50 words]

[List of numbered actions goes here, keeping original numbering]

Task#2: [start_index]~[end_index]  
Description: [Brief description of this specific subtask, max 50 words]

[List of numbered actions goes here, keeping original numbering]

Log content:
{log_content}"""

    try:
        print("GPT API 호출 중...")
        response = client.chat.completions.create(
            model="gpt-4o",  # 필요에 따라 모델 조정 가능
            messages=[
                {"role": "system", "content": "You are an expert at analyzing and labeling user interaction logs."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI API 호출 오류: {e}")
        return None

def process_logs(input_dir="action_logs", output_dir="labeled_logs"):
    """
    모든 미처리된 변환 로그 파일을 처리하여 라벨링합니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 미처리된 로그 파일 가져오기
    unlabeled_logs = get_unlabeled_log_files(input_dir, output_dir)
    
    if not unlabeled_logs:
        print("✅ 모든 로그 파일이 이미 라벨링되었습니다!")
        return True
    
    print(f"🔍 {len(unlabeled_logs)}개의 미처리된 로그 파일을 발견했습니다.")
    
    processed_count = 0
    
    for log_file in unlabeled_logs:
        output_file = generate_output_filename(log_file, output_dir)
        
        print(f"🔍 로그 파일 처리 중: {os.path.basename(log_file)}")
        
        # 로그 내용 읽기
        log_content = read_log_file(log_file)
        if not log_content:
            print(f"❌ 로그 내용을 읽을 수 없습니다: {log_file}")
            continue
        
        # GPT로 라벨링
        labeled_content = process_log_with_gpt(log_content)
        
        if labeled_content:
            # 라벨링 결과 저장
            try:
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(labeled_content)
                
                print(f"✅ 라벨링된 로그 저장 완료: '{os.path.basename(output_file)}'")
                processed_count += 1
            except Exception as e:
                print(f"❌ 파일 저장 오류: {e}")
        else:
            print(f"❌ 라벨링 실패: {os.path.basename(log_file)}")
        
        # API 속도 제한 회피를 위한 지연
        time.sleep(1.5)
    
    print(f"\n총 {processed_count}개의 로그 파일이 성공적으로 라벨링되었습니다.")
    return True

def check_api_key():
    """
    API 키가 설정되었는지 확인하고 안내합니다.
    """
    if not openai_api_key:
        print("⚠️ 경고: OpenAI API 키가 설정되지 않았습니다!")
        print("다음 명령으로 API 키를 설정하세요:")
        print("    openaikey.env 파일에 OPENAI_API_KEY=your_api_key_here 추가")
        return False
    return True

def main():
    """
    메인 함수 - 프로그램의 진입점
    """
    print("🔍 로그 자동 라벨링 프로그램을 시작합니다...")
    
    # API 키 확인
    if not check_api_key():
        return
    
    # 로그 처리
    process_logs()
    
    print("✅ 로그 라벨링 프로세스가 완료되었습니다.")

if __name__ == "__main__":
    main()