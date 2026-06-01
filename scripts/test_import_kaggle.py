import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_kaggle_env = os.path.join(ROOT_DIR, 'kaggle-environments-0.1.4')
print('ROOT_DIR=', ROOT_DIR)
print('local_kaggle_env=', local_kaggle_env)
print('sys.path before inserts:')
for p in sys.path[:5]:
    print('  ', p)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if local_kaggle_env not in sys.path:
    sys.path.insert(0, local_kaggle_env)
print('\nsys.path after inserts:')
for p in sys.path[:10]:
    print('  ', p)
try:
    import kaggle_environments
    print('\nimported kaggle_environments OK')
    import inspect
    print('version file exists:', os.path.exists(os.path.join(local_kaggle_env,'kaggle_environments','__init__.py')))
except Exception as e:
    print('\nimport failed:', type(e), e)
