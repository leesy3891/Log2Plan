# -*- coding: utf-8 -*-
# LocalPlanner.py
import os
import openai
from dotenv import load_dotenv
import re
import time
from GUI_Parser import GUI_Parser

load_dotenv('./openaikey.env')
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key:
    print("API 키가 성공적으로 설정되었습니다.")

client = openai.OpenAI(api_key=openai_api_key)

def parse_local_plan(input_string):
    # 'localPlan:' 이후 문자열 추출
    match = re.search(r'localPlan:(.*)', input_string)
    if not match:
        return None, None, None
    
    local_plan_str = match.group(1)

    # 'exc:'를 기준으로 분리
    parts = re.split(r'exc:', local_plan_str, maxsplit=1)
    if len(parts) < 2:
        return parts[0].strip(), None, None

    task_list = parts[0].strip()
    exc_part = parts[1].strip()

    # 'observation:' 이전까지를 exc 값으로 저장
    exc_match = re.match(r'(\w+)', exc_part)
    exc_value = exc_match.group(1) if exc_match else None

    # 'observation:' 이후 문자열 추출
    obsv_match = re.search(r'observation:(.*)', exc_part)
    obsv = obsv_match.group(1).strip() if obsv_match else None

    return task_list, exc_value, obsv

def localplan_pwa(idx, tasks, current_task, comp_dict, comp_type='pwa', task_context=None, task_substeps=None):    ### 실제로 전체 task가 아니라 tasks[idx-2:idx-3] 범위를 넣을 것. / command 도 입력으로 쓸 의향 있음  
# def localPlan(idx, task_list, task, comp_dict, comp_type='pwa', task_context=None, task_substeps=None):
    # task_list = [task[1] + ',' + task[2] for task in tasks] # localplan에서는 assist_bit 필요 없으므로 assist_bit제외한 task_list만 만들기 
    # [[0, 'open, Google Chrome']]
    task_lists = [task[1] + ', ' + task[2] for task in tasks if isinstance(task, list) and len(task) > 2]
    start_idx = 0 if idx < 2 else idx - 2
    end_idx = len(tasks)-1 if idx > len(tasks)-3 else idx + 3
    task_list = task_lists[start_idx:end_idx]

    print(idx ,": local plan pwa : ", task_list)
    print("current_task : ", current_task[1:])  
    
    if current_task[1].startswith('text input'):
        task_options = "1: click, text input field -> type, text -> press, enter / 2: click, text input field -> type, text -> press, tab"
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'text input'. Here are the available options to execute this task: 
        {task_options}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        Choose the best option and construct the task list using the available component dictionary (index: [component name, component type] form). If mouse action is needed, select the most semantically similar component to the object in task. Example: [click, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'search' ≈ '검색'), and select the component that best matches the task intent.
        Text input fields may appear as components with types such as 'Edit', 'TextField', or 'ComboBox'. Do not ignore 'ComboBox' if the label suggests text entry is possible.
        If the task mentions 'search bar', 'URL bar', or 'input field', match to components like 'Search Google or type a URL' or similar label phrases, even if full names differ.
        If 'press' action is possible, use PyAutoGUI's KEYBOARD_KEYS instead of a mouse click for faster execution.
        Always prefer shortcut keys over mouse actions when available (e.g., use 'f2' instead of clicking 'rename', 'ctrl+shift+n' for a new folder).

        Example:
        Task: ['text input', 'Seoul Station in Google search bar']
        Component: '8. Search Google or type a URL(ComboBox)'
        localPlan: [click, 8. Search Google or type a URL(ComboBox)], [type 'Seoul Station'], [press, enter]
        Reason: Component label matches the concept of a search input bar.

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=3000,
            temperature=0.5
        )
        response = response.choices[0].message.content

    # Scroll : next page dict 나눴기 때문에 수정해야함
    elif current_task[1].startswith('scroll'):
        task_options = "1: focus on, target window -> scroll, 1000 / 2: mousedown, scroll bar -> mouseup, target position"
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'scroll'. Here are the available options to execute this task: 
        {task_options}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        Choose the best option and construct the task list using the available component dictionary (index: [component name, component type] form). If mouse action is needed, select the most semantically similar component to the object in task. Example: [click, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent.
        If 'press' action is possible, use PyAutoGUI's KEYBOARD_KEYS instead of a mouse click for faster execution.
        Always prefer shortcut keys over mouse actions when available (e.g., use 'f2' instead of clicking 'rename', 'ctrl+shift+n' for a new folder).

        Example:
        Event: Scroll, for submit button
        localPlan: [mousedown, 56. Scroll(Scrollbar)], [mouseup, 37. submit(Button)]
        Reason: In PCS system, submit button is usually located at the bottom of the page. 

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=3000,
            temperature=0.5
        )
        
        response = response.choices[0].message.content

    elif current_task[1].startswith('open'):
        task_options = "1: press, win -> type, target -> press, enter / 2: doubleclick, target / 3: click, search bar -> type, target -> press, enter"
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'open'. Here are the available options to execute this task: 
        {task_options}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        Choose the best option and construct the task list using the available component dictionary (index: [component name, component type] form). If mouse action is needed, select the most suitable component including the index. Example: [Event, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent.
        For documents or image, pdf, etc., match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need').
        If 'press' action is possible, use PyAutoGUI's KEYBOARD_KEYS instead of a mouse click for faster execution.
        Always prefer shortcut keys over mouse actions when available (e.g., use 'f2' instead of clicking 'rename', 'ctrl+shift+n' for a new folder).

        Example:
        Event: Open, Notepad
        localPlan: [press win], [type 'Notepad'], [press enter]

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        response = response.choices[0].message.content
        
    elif current_task[1].startswith('close'):
        task_options = "1: focus on, title bar -> doubleclick, close button / 2: press, alt+f4" 
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'close'. Here are the available options to execute this task: 
        {task_options}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        Choose the best option and construct the task list using the available component dictionary (index: [component name, component type] form). If mouse action is needed, select the most suitable component including the index. Example: [close, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent.
        For documents or image, pdf, etc., match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need').
        If 'press' action is possible, use PyAutoGUI's KEYBOARD_KEYS instead of a mouse click for faster execution.
        Always prefer shortcut keys over mouse actions when available (e.g., use 'f2' instead of clicking 'rename', 'ctrl+shift+n' for a new folder).

        Example:
        Event: Close, Notepad
        localPlan: [press alt+f4]

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=3000,
            temperature=0.5
        )
        response = response.choices[0].message.content
    
    elif current_task[1].startswith('go-to'):
        task_options = "1: press ctrl+l -> type url -> press enter / 2: click hyperlink "
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'go to'. Here are the available options to execute this task: 
        {task_options}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        Choose the best option and construct the task list using the available component dictionary (index: [component name, component type] form). If mouse action is needed, select the most suitable component including the index. Example: [go to, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'Time & Language' ≈ '시간 및 언어'), and select the component that best matches the task intent.
        For hyperlinks or buttons, match using known titles or Web domain knowledge (e.g., 'page about Multihead Attention' → 'Attention is all you need!: A Paper Review').
        If 'press' action is possible, use PyAutoGUI's KEYBOARD_KEYS instead of a mouse click for faster execution.
        Always prefer shortcut keys over mouse actions when available (e.g., use 'f2' instead of clicking 'rename', 'ctrl+shift+n' for a new folder).

        Example:
        Event: Go To, Google homepage
        localPlan: [press, ctrl+l], [type, 'www.google.com'], [press, enter]

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        response = response.choices[0].message.content
    
    # 옵션 1개일 때
    elif current_task[1].startswith('click'):
        task_options = "click, target object"
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'click'. Here are the available option to execute this task: 
        {task_options}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        If component is in component dict (index: [component name, component type] form), select the most semantically similar component to the object in task. Example: [click, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent.
        For files or buttons, match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need').
        If 'press' action is possible, use PyAutoGUI's KEYBOARD_KEYS instead of a mouse click for faster execution.
        Always prefer shortcut keys over mouse actions when available (e.g., use 'f2' instead of clicking 'rename', 'ctrl+shift+n' for a new folder).

        Example:
        Event: Click, translate button
        localPlan: [click, 23. translate(MenuItem)]

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=3000,
            temperature=0.5
        )
        response = response.choices[0].message.content

    elif current_task[1].startswith('rightclick'):
        task_options = "rightclick, target object -> move focus, to child window(menu)"
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'rightclick'. Here are the available option to execute this task: 
        {task_options}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        If component is in component dict (index: [component name, component type] form), select the most semantically similar component to the object in task. Example: [rightclick, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent.
        For files or buttons, match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need').
        If 'press' action is possible, use PyAutoGUI's KEYBOARD_KEYS instead of a mouse click for faster execution.
        Always prefer shortcut keys over mouse actions when available (e.g., use 'f2' instead of clicking 'rename', 'ctrl+shift+n' for a new folder).

        Example:
        Event: Rightclick, UFO.docx  
        localPlan: [rightclick 45. UFO.docx(ListItem)], [move focus, Menu]

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        response = response.choices[0].message.content 

    elif current_task[1].startswith('drag'):
        task_option = "mousedown, target object -> mouseup, destination" 
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'drag'. Here are the available options to execute this task: 
        {task_option}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        If mouse action is needed and component is in component dict (index: [component name, component type] form), select the most semantically similar component to the object in task. Example: [click, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent.
        For documents or image, pdf, etc., match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need').
        If a keyboard shortcut is available for the same action (e.g., shift+pgdn for text selection), prefer it over a mouse drag.

        Example:
        Event: Drag, File_A to Google Lens  
        localPlan: [mousedown, 43. File_A(ListItem)], [mouseup, 15. Google Lens(Pane)]

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        response = response.choices[0].message.content

    elif current_task[1].startswith('copy'):
        task_option = "click, target -> press, ctrl+c"
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'copy'. Here are the available options to execute this task: 
        {task_option}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        If mouse action is needed and component is in component dict (index: [component name, component type] form), select the most semantically similar component to the object in task. Example: [click, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent.
        For documents or image, pdf, etc., match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need').
        If possible, prefer 'press' actions using PyAutoGUI's KEYBOARD_KEYS instead of mouse actions for faster execution.

        Example:
        Event: Copy, file about a fairytale  
        localPlan: [click, 15. Snow White(ListItem)], [press, ctrl+c]

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=3000,
            temperature=0.5
        )
        response = response.choices[0].message.content

    elif current_task[1].startswith('paste'):
        task_option = "click, target pane -> press, ctrl+v"
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'paste'. Here are the available options to execute this task: 
        {task_option}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        If mouse action is needed and component is in component dict (index: [component name, component type] form), select the most semantically similar component to the object in task. Example: [click, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent.
        Choose the area where a file can be pasted. Match using domain knowledge (e.g., '0325' → 'Attention is all you need').
        If possible, prefer 'press' actions using PyAutoGUI's KEYBOARD_KEYS instead of mouse actions for faster execution.

        Example:
        Event: Paste, Folder_Y  
        localPlan: [click 21. Folder_Y(Pane)], [press ctrl+v]

        Needed Inputs will be given by the user:
        """
        task_option = "click, target pane -> press, ctrl+v"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=3000,
            temperature=0.5
        )
        response = response.choices[0].message.content

    elif current_task[1].startswith('delete'):
        task_option = "click, target -> press, ctrl+d"
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'delete'. Here are the available options to execute this task: 
        {task_option}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        If mouse action is needed and component is in component dict (index: [component name, component type] form), select the most semantically similar component to the object in task. Example: [click, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent.
        For documents or image, pdf, etc., match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need').
        If possible, prefer 'press' actions using PyAutoGUI's KEYBOARD_KEYS instead of mouse actions for faster execution.

        Example:
        Event: Delete, File_Z  
        localPlan: [click, 15. File_Z(ListItem)], [press, ctrl+d]

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=3000,
            temperature=0.5
        )
        response = response.choices[0].message.content

    elif current_task[1].startswith('rename'):
        task_option = "click, target object -> press, f2 -> type, name -> press, enter"
        prompt = f"""
        You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env.
        Each task follows the format ['Event', 'object']. The object specifies where the action should take place.
        Event is 'rename'. Here are the available options to execute this task: 
        {task_option}

        Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions.
        If mouse action is needed and component is in component dict (index: [component name, component type] form), select the most semantically similar component to the object in task. Example: [click, index. component name(component type)]
        To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent.
        For documents or image, pdf, etc., match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need').
        Always prefer shortcut keys over mouse actions when available (e.g., 'f2' instead of right-clicking 'Rename').
        Use 'press f2' directly on the selected object to enter rename mode.

        Example:
        Event: Rename, Document_X  
        localPlan: [click, 14. Document_X(ListItem)], [press, f2], [type, NewName], [press, enter]

        Needed Inputs will be given by the user:
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<only the object delete all preposition>"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        response = response.choices[0].message.content
    elif current_task[1].startswith('save'):
        print(current_task)      # 제어 > press ctrl+s
        response = "local:[press, ctrl+s],exc:yes,observation:save the file"
    elif current_task[1].startswith('wait'):
        print(current_task)
        response = "local:[wait, screenchange],exc:yes,observation:wait for window script loading"
        # Detectchange 모듈 실행 (활성 윈도우 리스트 내 변화 또는 윈도우 이름 변화 트래킹)
    elif current_task[1].startswith('press'):
        print(current_task)      # 제어 > press keys
        response = f"local:[press, {current_task[2]}],exc:yes,observation:press the given shortcut keys."
    elif current_task[1].startswith('type'):
        print(current_task)     
        response = f"local:[type, {current_task[2]}],exc:yes,observation:type the given string."
    elif current_task[1].startswith('repeat'):
        # 숫자 범위만 추출 (3-8을 가져오기)
        match = re.search(r'(\d+)-(\d+)', current_task[2])
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            print(f"start: {start}, end: {end}")
        else:
            raise ValueError("올바른 숫자 범위를 찾을 수 없음")
        task_list = task_lists[start:end+1]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given a part of the task list, current task, and component dictionary extracted from Windows GUI env."},
                {"role": "system", "content": "Each task follows the format ['Event', 'object']. The object specifies where the action should take place."},
                {"role": "system", "content": "Event is 'repeat'. ['repeat', a-b, on (object)] means to modify the previous tasks a~b to on the given object."},
                {"role": "system", "content": "Use actions: click/rightclick/doubleclick/press/type/scroll to make a localPlan. Each task should take form of [action, object] with no prepositions."},
                {"role": "system", "content": "If mouse action is needed and component is in component dict(index: [component name, component type] form), select the most semantically similar component to the object in task. ex) [Event, index. component name(component type)]"},
                {"role": "system", "content": "To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent."},
                {"role": "system", "content": "For documents or image, pdf, etc.., match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need')."},
                {"role": "system", "content": "If 'press' action is possible, use PyAutoGUI's KEYBOARD_KEYS instead of a mouse click for faster execution."},
                {"role": "system", "content": "Always prefer shortcut keys over mouse actions when available (e.g., use 'f2' instead of clicking 'Rename', 'ctrl+shift+n' for a new folder)."},
                {"role": "system", "content": "Example:\nEvent: Repeat, 5-7, on IBM.png\nlocalPlan: [mousedown, 34. IBM.png(ListItem)], [switch focus, Chrome], [mouseup, Google Lens(Pane)]"},
                {"role": "user", "content": f"Task a-b: {task_list}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if Current task is executable (yes/no) with a reason in under 40 words. Answer 'no' only when there is no alternative to execute the task from current screen."},
                {"role": "user", "content": "Always format your response as follows -> locaPlan:<task list>,exc:<yes/no>, observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=400,
            temperature=0.4
        )
        response = response.choices[0].message.content           
        

    else:  # 정의되지 않은 Event
        task_options = "Split the task into defined actions: click, doubleclick, rightclick, press, type, drag. If no matching action exists, return an empty list."
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task."},
                {"role": "system", "content": "Each task follows the format ['Event', 'object']. The object specifies where the action should take place."},
                {"role": "system", "content": f"Event '{current_task[1]}' is not pre-defined. The available strategy is: {task_options}"},
                {"role": "system", "content": "Analyze the task and component dictionary to determine the best sequence of actions using only defined task actions (click, doubleclick, rightclick, press, type)."},
                {"role": "system", "content": "If mouse action is needed and component is in component dict(index: [component name, component type] form), select the most semantically similar component to the object in task. ex) [Event, index. component name(component type)]"},
                {"role": "system", "content": "To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent."},
                {"role": "system", "content": "For documents or image, pdf, etc.., match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need')."},
                {"role": "system", "content": "If 'press' action is possible, use PyAutoGUI's KEYBOARD_KEYS instead of a mouse click for faster execution."},
                {"role": "system", "content": "Always prefer shortcut keys over mouse actions when available (e.g., use 'F2' instead of clicking 'Rename', 'Ctrl+Shift+N' for a new folder)."},
                {"role": "system", "content": "If no valid mapping exists, return an empty list and indicate that the task cannot be executed."},
                {"role": "system", "content": "Example:\nEvent: 'send', 'message HI! to smwu_software'\nlocalPlan: [click, Send Message button], [type, HI!], [press, enter]"},
                {"role": "user", "content": f"{task_context}: {task_substeps}, Current task: {current_task}, Component dictionary: {comp_dict}."},
                {"role": "user", "content": "Provide the selected task list with the chosen component. Also, state if the action is executable (yes/no) with a reason in under 40 words."},
                {"role": "user", "content": "Always format your response as follows -> localPlan:<task list>,exc:<yes/no>,observation:<description of the current environment and reasoning>"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        response = response.choices[0].message.content

    print("--------")

    localplan, exc_value, obsv = parse_local_plan(response)

    print("LocalPlan:", localplan)
    print("Exc:", exc_value)
    print("Observation:", obsv)

    # 1. 대괄호 안의 문자열 추출
    matches = re.findall(r"\[([^\[\]]+?)\]", localplan)

    # 2. 각 항목을 분리 및 후처리
    localplan_list = []
    for item in matches:
        parts = item.strip().split(maxsplit=1)
        if len(parts) == 2:
            # 내용1 처리: 공백, ',' 제거 + 소문자 변환
            part1 = parts[0].replace(",", "").lower()
            # 내용2 처리: 앞뒤 공백 제거 + '', "" 제거
            part2 = parts[1].strip().strip("'\"")
            localplan_list.append([part1, part2])
        else:
            # 예외 처리 (예: 하나밖에 없는 경우 등)
            localplan_list.append([parts[0].replace(",", "").lower(), ""])

    print(f"localplan_list = {localplan_list}")

    return localplan_list, exc_value, obsv



# 실행되는 부분
def localPlan(idx, task_list, task, comp_dict, comp_type):
    if comp_type== 'pwa':
        # print("this is pwa")
        localplan, exc, obsv = localplan_pwa(idx, task_list, task, comp_dict)
    else:
        localplan = []
        exc = 'No'
        obsv = None
    return localplan, exc, obsv

#idx= 1
#task_list = [['0', 'open', 'RNDix folder'], ['0', 'copy', 'pdf about Multihead Attention'], ['0', 'open', '0325 folder'], ['0', 'paste', 'the paper on 0325'], ['0', 'rename', "paper as 'AIAYN'"]]
#task = ['0', 'copy', 'paper about Attention Mechanism']
#time.sleep(3)
#comp_dict, comp_type = GUI_Parser(f"./case0/{idx}.png")  # GUI 요소 탐색
#comp_dict_prompt = dict(list(comp_dict.items())[1:3])
#localPlan(idx, task_list, task, comp_dict_prompt, 'pwa')