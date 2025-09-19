from GlobalPlanner import globalPlanner, process_command
from UI_control import UI_Control
from NewPlanner import newPlan

def main():
    command = input("Automation Task: ")  
    content = process_command(command)
    # 글로벌 플래너 실행
    task_list = globalPlanner(command, content)
    print("Generated task_list:", task_list)
    print("글로벌 플래너 끝끝")
    incomplete_tasks, observations = UI_Control(task_list)  # UI_Control 실행-UI

    if incomplete_tasks:
        print("\n[Step 4] Handling Incomplete Tasks...")
        #new_task_list = newPlan(content, task_list, incomplete_tasks, observations)
        #UI_Control(new_task_list)  # 실패한 Task 재실행

if __name__ == "__main__":
    main()