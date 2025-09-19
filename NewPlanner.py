# New Planner.py
import os
import openai
from dotenv import load_dotenv

load_dotenv('.env')
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key:
    print("API 키가 성공적으로 설정되었습니다.")


client = openai.OpenAI(api_key=openai_api_key)

def newPlan(content, tasks, incomplete_tasks, obsv):
    #ChatCompletion.create(
    #chat.completions.create(
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You will be given a command and a task list that was made to execute the command. The task list is in a form of [[user_assist_bit, action, object], ...]."},
            {"role": "system", "content": "Some tasks in the task list were not completed, and you need to generate a new task list to complete these tasks from the current state. You can understand the current state by looking at the observation."},
            {"role": "system", "content": "The action of each task can be: click/double-click/right-click/press/type/drag/else. The object is where the action should be done."}, 
            {"role": "system", "content": "if 'action object' is 'press (black)', (blank) should be one or more KEYBOARD_KEYS in PyautoGUI."},
            {"role": "system", "content": "must choose using shortcut keys on Windows over click/right-click/doubleclick. ex:use 'f2' instead of 'click 'Rename', 'ctrl+shift+n' to create new folder, 'ctrl+g' to group selected objects.'" },
            {"role": "system", "content": "Each task is also associated with an user_assist_bit, which is a bit to tell if a specifical action should be done by the user or target for the task was not given. It should normally be 0, but should be 1 when user action is needed. ex: 'type user ID and PW', 'select objects to group', 'drag the area you want to copy', 'doubleclick the version that fits'"},
            {"role": "system", "content": f"The given command: {content}"},
            {"role": "system", "content": f"The initial task list: {tasks}"},
            {"role": "system", "content": f"The incomplete tasks were: {incomplete_tasks}."},
            {"role": "system", "content": f"the observation made from the current screen: {obsv}. Consider this when generating a new task list."},
            {"role": "system", "content": "You should generate a new task list so command can be executed correctly. use the observation to get a better understanding of the current situation."},
            {"role": "user", "content": "Generate a new task list to complete the command, considering the user_assist_bit for user intervention and the observation that caused the incomplete tasks. The new task list should be in the format [[user assist bit, action, object], ...] and must be different from the initial task list."}
        ],
        max_tokens=400,
        temperature=0.1
    )

    # Assuming the response contains the task list as required
    newplan = response.choices[0].message.content
    return eval(newplan)
