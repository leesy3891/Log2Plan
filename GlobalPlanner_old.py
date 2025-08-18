import os
import openai
from dotenv import load_dotenv
import re
import time
import numpy as np
import pickle
import glob

load_dotenv('./openaikey.env')
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key:
    print("API 키가 성공적으로 설정되었습니다.")

from openai import OpenAI

client = openai.OpenAI()

def read_labeled_logs():
    """
    labeled_logs 디렉토리에서 모든 로그 파일을 읽어옵니다.
    """
    logs = []
    # labeled_logs 디렉토리의 모든 .txt 파일 검색
    for file_path in glob.glob('labeled_logs/*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logs.append({
                'file_path': file_path,
                'content': content
            })
    return logs

def extract_task_units(logs):
    """
    로그 파일에서 작업 단위(Task Unit)를 추출합니다.
    """
    task_units = []
    
    for log in logs:
        content = log['content']
        file_name = os.path.basename(log['file_path'])
        
        # 정규식을 사용하여 Task Unit 추출
        task_unit_pattern = r'Task Unit #\d+: \*\*(.*?)\*\*\s+Description: (.*?)(?=Task#\d+:|$)'
        task_units_matches = re.findall(task_unit_pattern, content, re.DOTALL)
        
        for i, (name, description) in enumerate(task_units_matches):
            unit_num = i + 1
            unit_id = f"{file_name}:Unit{unit_num}"
            
            # 해당 Task Unit의 모든 내용 추출 (Task들과 서브태스크 포함)
            unit_start = content.find(f"Task Unit #{unit_num}")
            next_unit = f"Task Unit #{unit_num + 1}"
            unit_end = content.find(next_unit) if next_unit in content else len(content)
            unit_content = content[unit_start:unit_end].strip()
            
            task_units.append({
                'id': unit_id,
                'name': name.strip(),
                'description': description.strip(),
                'content': unit_content
            })
    
    return task_units

def create_embeddings(texts):
    """
    텍스트 리스트에 대한 임베딩을 생성합니다.
    """
    batch_size = 20  # API 호출 횟수 줄이기 위한 배치 크기
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def create_vector_db():
    """
    작업 단위의 벡터 데이터베이스를 생성합니다.
    """
    db_file = "task_unit_vectors.pkl"
    
    # 이미 생성된 데이터베이스가 있으면 로드
    if os.path.exists(db_file):
        with open(db_file, 'rb') as f:
            return pickle.load(f)
    
    # 로그 파일 읽기
    logs = read_labeled_logs()
    print(f"{len(logs)}개의 로그 파일을 읽었습니다.")
    
    # 작업 단위 추출
    task_units = extract_task_units(logs)
    print(f"{len(task_units)}개의 작업 단위를 추출했습니다.")
    
    # 임베딩 생성할 텍스트 준비: 이름 + 설명
    texts_to_embed = [f"{unit['name']}\n{unit['description']}" for unit in task_units]
    
    # 임베딩 생성
    print("임베딩을 생성 중입니다...")
    embeddings = create_embeddings(texts_to_embed)
    
    # 벡터 데이터베이스 구성
    vector_db = []
    for i, unit in enumerate(task_units):
        vector_db.append({
            'id': unit['id'],
            'name': unit['name'],
            'description': unit['description'],
            'content': unit['content'],
            'embedding': embeddings[i]
        })
    
    # 벡터 데이터베이스 저장
    with open(db_file, 'wb') as f:
        pickle.dump(vector_db, f)
    
    print(f"벡터 데이터베이스가 {db_file}에 저장되었습니다.")
    return vector_db

def search_similar_tasks(query, vector_db, top_k=3):
    """
    쿼리와 유사한 작업 단위를 검색합니다.
    """
    # 쿼리 임베딩 생성
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[query]
    )
    query_embedding = response.data[0].embedding
    
    # 유사도 계산
    similarities = []
    for item in vector_db:
        similarity = np.dot(query_embedding, item['embedding'])
        similarities.append((item, similarity))
    
    # 유사도 기준 정렬 및 상위 결과 반환
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = [item for item, _ in similarities[:top_k]]
    
    return top_results

def globalPlanner(content):
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m" 
    # 벡터 데이터베이스 로드 또는 생성
    vector_db = create_vector_db()
    
    # 유사한 작업 검색
    print("유사한 작업을 검색합니다...")
    similar_tasks = search_similar_tasks(content, vector_db, top_k=3)
    
    # 검색된 작업 내용 포맷팅
    examples = [task['content'] for task in similar_tasks]
    examples_text = "\n\nNEXT:\n".join(examples)
    print(f"{CYAN}Examples to show to the prompt: {examples_text}{RESET}")
    
    print(f"{len(similar_tasks)}개의 유사한 작업을 찾았습니다.")
    
    # 자동화 계획 생성
    print("자동화 계획을 생성합니다...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "you are a Desktop automation agent GUI control automation task. your OS is Windows 10. make a list of subtasks which is a combination of the given Events to fulfill the given task." },
            {"role": "system", "content": "Each subtask is in 'assist bit# Event, objects' form. User assist bit is a bit to tell if a Event should be done by the user or target for the task was not given. It should normally be 0, but should be 1 when user action is needed. ex: 'input user ID and PW', 'drag the area you want to copy', 'doubleclick the version that fits'. object is the target of the action." },
            {"role": "system", "content": "Given Events(#18): click/rightclick/drag/scroll/press/text input/open/close/switch focus/go-to/save/copy/paste/delete/rename/login/repeat/wait.  ex: 0#switch focus, back to Excel sheet, 0#click, Image button" },
            {"role": "system", "content": "As an exception, when the Event is repeat, the subtask should be in a form of 'assist bit# repeat, task#a-b, on object_A' like '9. 0#repeat, 3-8, on fileB, 10. 0#repeat, 3-8, on fileC...'. Event: switch focus is when focused window needs change(not needed when focus is naturally changed by the previous Event, like opening Chrome). Event: go-to means navigation to another website inside current window.(not needed when navigation happens by the previous Event, like clicking a hyperlink)" },
            {"role": "system", "content": "if subtask is 'press (black)', (blank) should be KEYBOARD_KEYS in PyautoGUI." }, 
            {"role": "system", "content": "If possible, choose using shortcut keys on Windows over click/rightclick/doubleclick. ex:use 'f2' instead of 'click 'Rename', 'ctrl+shift+n' to create new folder, 'ctrl+g' to group selected objects.' Use Chrome as your browser on Web tasks, and open files/apps directely using without traversing through file explorer. (EX: 1. 0#press, win\n2. 0#text input, Filename.txt\n 3. press, enter)" },
            {"role": "system", "content": "Here is one example, Example: execute kakaotalk -> (1. 0#open, kakaotalk\n2. 1#login, on kakaotalk)" }, 
            {"role": "system", "content": "You are an automation planner AI. You should take inspiration from past task approach. Do NOT just copy and be creative. "},
            {"role": "system", "content": "Format your response only as a list of subtasks without task descriptions"},
            ##{"role": "system", "content": "If the `target_name` ends with parentheses (e.g., `(Text)`, `(Button)`, `(Navigation)`), you must **remove the entire parentheses and its contents**. Only output the clean target name without control type."},
            {"role": "user", "content": f"Here are some related past Task for inspiration (these are just examples, not exact solutions):\n{examples_text}"},
            {"role": "user", "content": f"Task: {content}"}
        ],
        max_tokens=4000,
        temperature=0.4
    )

    print(f"Given command: {content}")
    output = response.choices[0].message.content           
    print(f"{MAGENTA}Task Splitted: {output}{RESET}")  

    output = tokenizer(output)
    print(output)
    return output    ## 굳이 모듈로 만들어 사용할 필요?

def tokenizer(output):
    parsed_list = []
    for line in output.split("\n"):
        # 정규식으로 task 패턴 파싱
        match = re.match(r"(\d+)\.\s*(\d+)#([^,]+),\s*(.+)", line)
        if match:
            assist_bit, action, obj = match.group(2).strip(), match.group(3).strip(), match.group(4).strip()
            parsed_list.append([assist_bit, action, obj])
        else:
            print(f"Warning: Unrecognized task format - {line}")
    
    return parsed_list

command = input("Command: ")
print(f"Given command: {command}")
print(f"content: {command}")
task_list = globalPlanner(command)