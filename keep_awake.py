import ctypes
import time

print("Preventing Windows from going to sleep. Press Ctrl+C to exit.")

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
# 0x80000000 | 0x00000001 | 0x00000002 = 0x80000003
ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Restoring original execution state...")
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
