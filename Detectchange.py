import time
from pywinauto import Desktop
from win32gui import GetForegroundWindow
from pywinauto.controls.hwndwrapper import HwndWrapper
from threading import Thread, Event


def find_new_window(timeout=5, stop_event=None):
    """
    새로운 윈도우를 감지하고 타임아웃 안에 포커스를 설정.
    :param timeout: 타임아웃 시간 (초)
    :param stop_event: 스레드 종료를 제어하는 이벤트
    """
    current_windows = {w.window_text(): w for w in Desktop(backend="uia").windows()}
    print("새로운 윈도우 감지를 시작합니다...")

    while timeout > 0 and not stop_event.is_set():
        time.sleep(0.8)
        timeout -= 0.8

        # 현재 활성 창 목록 확인
        new_windows = {w.window_text(): w for w in Desktop(backend="uia").windows()}
        added = set(new_windows) - set(current_windows)

        if added:
            for window_name in added:
                print(f"새로운 윈도우 활성화: {window_name}")
                try:
                    new_windows[window_name].set_focus()
                    print(f"{window_name} 창에 포커스 설정 완료.")
                except Exception as e:
                    print(f"{window_name} 창에 포커스 설정 실패: {e}")
            return True  # 새로운 창 감지 및 처리 완료

    print("새로운 창을 감지하지 못했습니다. 타임아웃 초과.")
    return False


def get_window_info():
    """현재 포커스된 창 정보를 반환."""
    try:
        hwnd = GetForegroundWindow()  # 현재 활성 창의 핸들 가져오기
        if hwnd:
            window = HwndWrapper(hwnd)
            return {
                'title': window.window_text(),
                'hwnd': hwnd,
                'class_name': window.friendly_class_name(),
            }
    except Exception as e:
        return {'error': str(e)}


def monitor_window_changes(interval=0.8, timeout=6, stop_event=None):
    """포커스된 창의 변화를 감지."""
    previous_window = None
    start_time = time.time()

    print("Monitoring focused window changes. Press Ctrl+C to stop.")

    while not stop_event.is_set():
        current_window = get_window_info()

        if not current_window:
            print("No focused window detected.")
        elif previous_window != current_window:
            print("Window changed:")
            for key, value in current_window.items():
                print(f"  {key}: {value}")

            previous_window = current_window
            return True  # 창 변화 감지

        # 타임아웃 체크
        if time.time() - start_time > timeout:
            print("Timeout: No window change detected within the timeout period.")
            return False

        time.sleep(interval)

    print("Stopped monitoring window changes.")
    return False


def detect_changes():
    result = False
    stop_event = Event()  # 이벤트 플래그 생성

    def run_find_new_window():
        nonlocal result
        if find_new_window(timeout=5, stop_event=stop_event):
            print("find_new_window detected a change.")
            result = True
            stop_event.set()  # 다른 스레드 종료 요청

    def run_monitor_window_changes():
        nonlocal result
        if monitor_window_changes(interval=0.8, timeout=5, stop_event=stop_event):
            print("monitor_window_changes detected a change.")
            result = True
            stop_event.set()  # 다른 스레드 종료 요청

    thread1 = Thread(target=run_find_new_window)
    thread2 = Thread(target=run_monitor_window_changes)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    if result:
        print("Result: True")
    else:
        print("Timeout occurred without detecting any changes.")

    return result
