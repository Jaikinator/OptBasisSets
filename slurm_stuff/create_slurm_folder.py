import os
import sys

if __name__ == "__main__":
    if not os.path.exists(sys.argv[1]):
        os.mkdir(sys.argv[1])