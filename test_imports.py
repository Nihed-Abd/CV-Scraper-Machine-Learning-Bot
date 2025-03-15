import sys
import traceback

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        print(f"❓ Other error importing {module_name}: {e}")
        return False

# Test all required modules
modules = [
    "telegram",
    "pandas",
    "reportlab",
    "fpdf",
    "beautifulsoup4",
    "bs4",
    "requests",
    "selenium",
    "matplotlib",
    "sklearn",
    "numpy",
    "logging",
    "datetime",
    "json",
    "os",
    "random",
    "time"
]

print("Testing imports...")
for module in modules:
    test_import(module)

# Now try to import from our modules
print("\nTesting project modules...")
try:
    import bot_modules
    print("✅ Successfully imported bot_modules")
    
    try:
        from bot_modules import scraper
        print("✅ Successfully imported bot_modules.scraper")
    except Exception as e:
        print(f"❌ Failed to import scraper: {e}")
        
    try:
        from bot_modules import data_processor
        print("✅ Successfully imported bot_modules.data_processor")
    except Exception as e:
        print(f"❌ Failed to import data_processor: {e}")
        
    try:
        from bot_modules import ml_models
        print("✅ Successfully imported bot_modules.ml_models")
    except Exception as e:
        print(f"❌ Failed to import ml_models: {e}")
        
except Exception as e:
    print(f"❌ Failed to import bot_modules: {e}")

print("\nDone testing imports.")
