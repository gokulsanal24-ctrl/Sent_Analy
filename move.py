import shutil
import os

try:
    if os.path.exists('templates/static'):
        shutil.move('templates/static', 'static')
    if os.path.exists('index.html'):
        os.remove('index.html')
    print("Files moved and cleaned up successfully.")
except Exception as e:
    print(f"Error: {e}")
