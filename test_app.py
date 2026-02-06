import sys
sys.path.insert(0, '.')
try:
    from app_enhanced import main
    from app import main as app_main
    print("All modules imported successfully!")
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
