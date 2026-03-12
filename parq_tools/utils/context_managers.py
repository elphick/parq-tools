import sys
import threading
import time
from datetime import datetime, timedelta

class timed:
    def __init__(self, process_name):
        self.process_name = process_name
        self._spinner_running = False
        self._spinner_thread = None

    def _spinner(self):
        spinner_chars = '|/-\\'
        idx = 0
        sys.stdout.write(f"{self.process_name}... ")
        sys.stdout.flush()
        while self._spinner_running:
            sys.stdout.write(spinner_chars[idx % len(spinner_chars)])
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b')
            idx += 1

    def __enter__(self):
        self.start_time = datetime.now()
        self._spinner_running = True
        self._spinner_thread = threading.Thread(target=self._spinner)
        self._spinner_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._spinner_running = False
        self._spinner_thread.join()
        end_time = datetime.now()
        elapsed = end_time - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))
        sys.stdout.write(f"\bDone. Elapsed: {elapsed_str}\n")
        sys.stdout.flush()



if __name__ == '__main__':
    with timed("Processing data"):
        # your code here
        time.sleep(5)