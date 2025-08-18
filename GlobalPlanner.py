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

def extract_task_unit_headers(gpt_output: str) -> str:
    """
    process_command() 가 돌려준 GPT 출력에서
    'Task Unit : **ENV[...] ACT[...] ...**' 줄만 골라서
    한 줄의 문자열(여러 개면 줄바꿈으로 연결)로 반환한다.
    """
    pattern = r'Task\s*Unit\s*:?\s*(\*\*.*?\*\*)'
    headers = re.findall(pattern, gpt_output)
    if not headers:
        print("⚠ No Task Unit headers matched.")

    return "\n".join([f"Task Unit : {h}" for h in headers])


# ---------- 개별 Task# 블록 추출 ----------
def extract_tasks(unit_text: str):
    """
    Task Unit   문자열 안에서  Task#n 블록들을 전부 잘라 리스트로 반환.
    반환 예: ['Task#1: ...', 'Task#2: ...', ...]
    """
    pattern = r'(Task#\d+:[\s\S]*?)(?=\nTask#\d+:|$)'
    return [m.group(1).strip() for m in re.finditer(pattern, unit_text)]


# ------------------------------------------------

def extract_task_units(logs):
    """
    로그 파일에서 작업 단위(Task Unit)를 추출하고, ENV, ACT, Title을 분리하여 포함합니다.
    
    ->>> 여기 수정함!! 
    """
    task_units = []

    for log in logs:
        content   = log['content']
        file_name = os.path.basename(log['file_path'])

        # Task Unit 블록 추출
        pattern = r'Task Unit #\d+: \*\*(.*?)\*\*\s+Description: (.*?)(?=Task#\d+:|$)'
        matches = re.findall(pattern, content, re.DOTALL)

        for idx, (name, description) in enumerate(matches, start=1):
            unit_id      = f"{file_name}:Unit{idx}"
            unit_start   = content.find(f"Task Unit #{idx}")
            next_header  = f"Task Unit #{idx + 1}"
            unit_end     = content.find(next_header) if next_header in content else len(content)
            unit_content = content[unit_start:unit_end].strip()

            env_tags, act_tag, title = parse_tags(name)

            task_units.append({
                'id'        : unit_id,
                'file_path' : log['file_path'],
                'name'      : name.strip(),
                'description': description.strip(),
                'content'   : unit_content,
                'env_tag'   : env_tags,      # 리스트 or 문자열
                'act_tag'   : act_tag,
                'title'     : title
            })

    return task_units


def parse_tags(name): 
    """
    ENV[], ACT[], Title을 Task Unit 헤더에서 분리.
    - ENV는 여러 개일 수 있어 리스트로 반환
    - Title에서 괄호로 끝나는 범위(예: (0~31)) 제거
    """
    env_match = re.search(r'ENV\[(.*?)\]', name)
    act_match = re.search(r'ACT\[(.*?)\]', name)

    env_raw  = env_match.group(1).strip() if env_match else ''
    env_tags = [e.strip() for e in env_raw.split(',')] if env_raw else []

    act_tag = act_match.group(1).strip() if act_match else ''

    # Title = ACT] 뒤에 오는 부분
    title_part = re.sub(r'.*ACT\[[^\]]*\]\s*', '', name).strip()
    # 괄호로 끝나는 범위 제거
    title = re.sub(r'\s*\(.*?\)\s*$', '', title_part).strip()

    return env_tags, act_tag, title


def create_embeddings(texts):
    """
    텍스트 리스트에 대한 임베딩을 생성합니다.
    """
    batch_size = 20  # API 호출 횟수 줄이기 위한 배치 크기
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def log_result(query, similar_units, output, llm_output, log_dir="result_logs"):
    """
    생성 과정과 결과를 로그 파일로 저장합니다.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 타임스탬프 -> 로그 파일명 생성
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"result_{timestamp}.txt")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Query: {query}\n\n")
        
        # 유사한 Task Unit 정보
        f.write(f"Similar Task Units ({len(similar_units)}):\n")
        for i, unit in enumerate(similar_units):
            f.write(f"{i+1}. {unit['id']} - ENV[{unit['env_tag']}] ACT[{unit['act_tag']}] Title[{unit.get('title', '')}] (Similarity: {unit.get('similarity', 0):.4f})\n")
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

def create_vector_db():
    """
    (1) labeled_logs의 모든 Task Unit을 파싱해 ENV / ACT / Title 정보를 포함시키고  
    (2) OpenAI 임베딩으로 벡터를 만든 뒤  
    (3) task_unit_vectors.pkl 로 저장·로드하는 헬퍼.

    ->>> 위랑 호환되게 수정
    """
    db_file = "task_unit_vectors.pkl"

    # 이미 만들어 둔 DB가 있으면 바로 반환
    if os.path.exists(db_file):
        with open(db_file, "rb") as f:
            return pickle.load(f)

    # 1) 로그 읽기 & Task Unit 추출
    logs = read_labeled_logs()
    print(f"{len(logs)}개의 로그 파일을 읽었습니다.")

    task_units = extract_task_units(logs)
    print(f"{len(task_units)}개의 작업 단위를 추출했습니다.")

    #  2) 임베딩 생성
    #    ENV는 리스트이므로 문자열로 합쳐 줌
    texts_to_embed = [
        f"{', '.join(unit['env_tag'])}\n{unit['act_tag']}"
        for unit in task_units
    ]

    print("임베딩을 생성 중입니다...")
    embeddings = create_embeddings(texts_to_embed)

    # 3) 벡터 DB 구성
    vector_db = []
    for i, unit in enumerate(task_units):
        vector_db.append({
            "id"         : unit["id"],
            "file_path"  : unit["file_path"],
            "content"    : unit["content"],
            "env_tag"    : unit["env_tag"],      # list 형태 유지
            "act_tag"    : unit["act_tag"],
            "title"      : unit["title"],
            "description": unit["description"],
            "embedding"  : embeddings[i]
        })

    # 4) 저장
    with open(db_file, "wb") as f:
        pickle.dump(vector_db, f)

    print(f"벡터 데이터베이스가 {db_file}에 저장되었습니다.")
    return vector_db


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)  # 0 나눗셈 방지


def search_diverse_similar_tasks(query: str, vector_db: list, top_k: int = 9):
    """
    ─ 단계적 다이버시티 선택 로직 ─
    ① 쿼리와 가장 비슷한 순으로 정렬
    ② 상위 3개 선택
    ③ 선택된 각 항목과 '가장 비슷한' 후보 1개씩을 제외한 뒤
       다시 상위 3개 → 총 6개
    ④ 방금 뽑은 3개에 대해 같은 방식으로 1개씩 제외하고
       다시 상위 3개 → 총 9개
    """
    # 1) 쿼리 임베딩
    query = query.strip()
    if not query:
        raise ValueError("❌ Provided query is empty. Cannot create embedding.")

    query_emb = np.array(
        client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        ).data[0].embedding
    )

    # 2) 유사도 계산 및 정렬
    scored = []
    for idx, item in enumerate(vector_db):
        sim = cosine_similarity(query_emb, item['embedding'])
        scored.append((idx, sim))
    scored.sort(key=lambda x: x[1], reverse=True)

    selected, excluded = [], set()

    def exclude_most_similar(to_idx):
        """to_idx 와 가장 비슷한 단일 항목의 인덱스를 excluded 집합에 추가"""
        base_emb = np.array(vector_db[to_idx]['embedding'])
        best_j, best_sim = None, -1
        for j, _ in scored:
            if j in excluded or j == to_idx:
                continue
            s = cosine_similarity(base_emb, vector_db[j]['embedding'])
            if s > best_sim:
                best_sim, best_j = s, j
        if best_j is not None:
            excluded.add(best_j)

    stage_targets = [3, 3, 3]      # 3-3-3
    for stage_size in stage_targets:
        for idx, _ in scored:
            if idx in excluded:
                continue
            selected.append(idx)
            excluded.add(idx)
            exclude_most_similar(idx)   # 가장 비슷한 것 1개 제외
            if len(selected) % stage_size == 0:
                break

    # top_k 초과 시 앞에서부터 자름
    selected = selected[:top_k]
    return [vector_db[i] for i in selected]


# ---------- NEW: 각 Task#별 유사 Task# 2개씩 뽑기 ----------
def top_similar_tasks_per_current(content_text: str,
                                  candidate_units: list,
                                  top_n: int = 1):
    """
    content_text  (process_command 결과) 안의 Task#마다
    candidate_units (search_diverse_similar_tasks 로 얻은 Task Unit 리스트) 안의
    Task# 들 중 cosine 기준 상위 top_n 개 선택.
    반환: 선택된 Task# 문자열 리스트 (순서는 content Task#1 → Task#2 …)
    """
    # ① 현재 명령의 Task# 분해
    cur_tasks = extract_tasks(content_text)
    if not cur_tasks:
        return []   # 예외 처리

    # ② 후보 Task# 전부 분해 & 보존
    cand_tasks = []
    for unit in candidate_units:
        for t in extract_tasks(unit['content']):
            cand_tasks.append(t)

    # ③ 임베딩 한꺼번에 생성  (API 호출 최소화)
    all_texts = cur_tasks + cand_tasks
    all_embs = create_embeddings(all_texts)
    cur_embs = all_embs[:len(cur_tasks)]
    cand_embs = all_embs[len(cur_tasks):]

    # ④ Task별 top-n 선정
    selected = []
    for i, cur_emb in enumerate(cur_embs):
        sims = []
        for j, cand_emb in enumerate(cand_embs):
            sim = cosine_similarity(cur_emb, cand_emb)
            sims.append((sim, cand_tasks[j]))
        sims.sort(key=lambda x: x[0], reverse=True)
        top_examples = [t for _, t in sims[:top_n]]
        selected.extend(top_examples)

    return selected


# --------------------------------------------------------------

def process_command(command):
    """
    GPT API를 사용하여 로그 내용을 처리합니다.
    """
    # 프롬프트 템플릿
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
            model="gpt-4o",  # 필요에 따라 모델 조정 가능
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


def globalPlanner(command, content):
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m" 
    # 벡터 데이터베이스 로드 또는 생성
    vector_db = create_vector_db()
    
    # 유사한 작업 검색
    print("유사한 작업을 검색합니다...")
    query_text = extract_task_unit_headers(content)
    
    similar_tasks = search_diverse_similar_tasks(query_text, vector_db, top_k=9)
    print(f"{len(similar_tasks)}개의 유사한 작업을 찾았습니다.")

    examples_list = top_similar_tasks_per_current(content, similar_tasks, top_n=2)
    examples_text = "\n\n".join(examples_list)
    print(f"{CYAN}Examples to show to the prompt: {examples_text}{RESET}")
    
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
            {"role": "system", "content": "You are an automation planner AI. You should take inspiration from past task approach. The given data has some noises in Event: Switch focus/drag which should have been click, object. So do NOT just copy and generate a Executable task sequence."},
            {"role": "system", "content": "Format your response only as a list of subtasks without task descriptions"},
            {"role": "user", "content": f"Here are some related past Task for inspiration (these are just examples, not exact solutions):\n{examples_text}"},
            {"role": "user", "content": f"Task: {command}"}
        ],
        max_tokens=4000,
        temperature=0.4
    )
    print(f"Given command: {command}")
    output = response.choices[0].message.content            
    #output2 = "1. 0#open, chrome\n2. 0#go to, https://chatgpt.com. 0#click, Upload files and more(Button)\n4. 0#click, Upload from computer(MenuItem)\n5. 0#switch focus, File explorer\n6. 0#click, a paper about Multihead Attention(ListItem)\n7. 0#press, enter\n8. 0#text input, 'Give me a summary of this paper.' \n9. 0#wait, for the answer"
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