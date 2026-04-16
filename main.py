import subprocess
import sys
import os

def main():
    env = os.environ.copy()
    root = os.path.dirname(os.path.abspath(__file__))
    numbers_dir = os.path.join(root, "Numbers")
    
    # Add both root and Numbers to PYTHONPATH
    env["PYTHONPATH"] = root + os.pathsep + numbers_dir
    
    #subprocess.run([sys.executable, "Numbers/numbers_classification.py"], env=env)

    iris_dir = os.path.join(root, "Iris")
    env["PYTHONPATH"] = root + os.pathsep + iris_dir

    subprocess.run([sys.executable, "Iris/scripts/iris_classification.py"], env=env)

if __name__ == "__main__":
    main()