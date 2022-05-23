import time
import os

if __name__ == "__main__":

    t = time.time()
    with open("./a_small_file.txt", 'a') as f:
        f.write('hello')
    print("append to a small file:", time.time() - t)

    t = time.time()
    with open("./joint_cmd.txt", 'a') as f:
        f.write('hello')
    print("append to a large file:", time.time() - t)
