import subprocess
import sys
import os

def main():
    env = os.environ.copy()
    root = os.path.dirname(os.path.abspath(__file__))
    iris_dir = os.path.join(root, "Iris")
    numbers_dir = os.path.join(root, "Numbers")
    env["PYTHONPATH"] = os.pathsep.join([root, iris_dir, numbers_dir])
    subprocess.run([sys.executable, "Iris/scripts/iris_classification.py"], env=env)
    subprocess.run([sys.executable, "Numbers/numbers_classification.py"], env=env)


if __name__ == "__main__":
    main()