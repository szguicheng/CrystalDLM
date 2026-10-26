"""Write the checked comparison after both independent pipelines have exited."""
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
while not all((root/f'{arm}-exit.txt').exists() for arm in ['ar','flare']):
    time.sleep(30)
subprocess.run([sys.executable,str(root/'summarize.py'),'--allow-partial'],check=True)
