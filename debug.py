import os
import sys

print("=== Complete dotenv Debug ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Show file existence and contents
env_file = '.env'
print(f"\n--- File Check ---")
print(f"File exists: {os.path.exists(env_file)}")
print(f"File path: {os.path.abspath(env_file)}")

if os.path.exists(env_file):
    with open(env_file, 'rb') as f:  # Read as bytes to see encoding
        raw_content = f.read()
        print(f"Raw bytes: {raw_content}")
    
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"UTF-8 content: '{content}'")
        print(f"Content length: {len(content)}")
        print(f"Content repr: {repr(content)}")

# Before loading dotenv
print(f"\n--- Before dotenv ---")
print(f"Test from os.getenv: {os.getenv('Test')}")
print(f"Test from os.environ.get: {os.environ.get('Test')}")
print(f"All env vars containing 'Test': {[k for k in os.environ.keys() if 'Test' in k]}")

# Try importing dotenv
try:
    from dotenv import load_dotenv, find_dotenv
    print(f"\n--- Loading dotenv ---")
    
    # Method 1: find_dotenv
    dotenv_file = find_dotenv()
    print(f"find_dotenv() found: {dotenv_file}")
    result1 = load_dotenv(dotenv_file)
    print(f"load_dotenv(find_dotenv()) returned: {result1}")
    
    # Method 2: explicit path
    result2 = load_dotenv('.env')
    print(f"load_dotenv('.env') returned: {result2}")
    
    # Method 3: default
    result3 = load_dotenv()
    print(f"load_dotenv() returned: {result3}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Manual parsing fallback
    print("--- Manual parsing fallback ---")
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
                    print(f"Manually set: {key} = {value}")

# After loading
print(f"\n--- After dotenv ---")
print(f"Test from os.getenv: {os.getenv('Test')}")
print(f"Test from os.environ.get: {os.environ.get('Test')}")
print(f"Test from os.environ dict: {os.environ.get('Test', 'NOT_FOUND')}")
print(f"All env vars containing 'Test': {[k for k in os.environ.keys() if 'Test' in k]}")

# Show all environment variables (first 10)
print(f"\n--- Sample environment variables ---")
env_items = list(os.environ.items())[:10]
for key, value in env_items:
    print(f"{key} = {value[:50]}..." if len(value) > 50 else f"{key} = {value}")

print("=== End Debug ===")