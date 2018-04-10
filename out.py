from pynput.keyboard import Key, Controller
import time

keyboard = Controller()
i = 0
while True:
    with open("buffer.txt", "r") as f:
        s = f.readlines()
        s = list(s)
        s = s[i]
        keyboard.press(s)
        time.sleep(1)
        keyboard.release(s)
        i += 1
