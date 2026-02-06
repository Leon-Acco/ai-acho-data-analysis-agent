#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

import sys

def test_imports():
    """Test all critical imports for the application."""
    print("Testing imports for AI Data Analysis Agent...")
    
    # Test basic imports
    try:
        import os
        import re
        from pathlib import Path
        print("✅ Standard library imports OK")
    except ImportError as e:
        print(f"❌ Standard library import error: {e}")
        return False
    
    # Test third-party imports
    third_party_modules = [
        ("duckdb", "DuckDB database"),
        ("pandas", "Pandas data analysis"),
        ("streamlit", "Streamlit web framework"),
        ("dotenv", "Python dotenv"),
        ("openai", "OpenAI client"),
    ]
    
    for module_name, description in third_party_modules:
        try:
            __import__(module_name)
            print(f"✅ {description} import OK")
        except ImportError as e:
            print(f"❌ {description} import error: {e}")
            return False
    
    # Test our own modules
    print("\nTesting application module imports...")
    app_modules = [
        ("agent_core", "build_agent"),
        ("agent_enhanced", "build_advanced_agent"),
        ("duckdb_manager", "DuckDBManager"),
    ]
    
    for module_name, function_name in app_modules:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name} import OK")
            
            # Try to import specific function if requested
            if function_name:
                try:
                    getattr(module, function_name)
                    print(f"  ✅ Function {function_name} available")
                except AttributeError:
                    print(f"  ⚠️  Function {function_name} not found in module")
        except ImportError as e:
            print(f"❌ {module_name} import error: {e}")
            return False
    
    # Test Agno specific imports
    print("\nTesting Agno framework imports...")
    try:
        from agno.agent import Agent
        from agno.models.deepseek import DeepSeek
        print("✅ Agno framework imports OK")
    except ImportError as e:
        print(f"❌ Agno import error: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True

def test_app_specific():
    """Test imports that are specific to our application."""
    print("\nTesting application-specific imports...")
    
    # Test imports from agent_core
    try:
        from agent_core import (
            TableInfo,
            get_table_info,
            render_schema_context,
            build_agent,
            extract_sql
        )
        print("✅ agent_core imports OK")
    except ImportError as e:
        print(f"❌ agent_core import error: {e}")
        return False
    
    # Test imports from agent_enhanced
    try:
        from agent_enhanced import (
            TableInfo as EnhancedTableInfo,
            get_table_info_with_samples,
            validate_sql,
            generate_analysis_summary,
            suggest_visualizations
        )
        print("✅ agent_enhanced imports OK")
    except ImportError as e:
        print(f"❌ agent_enhanced import error: {e}")
        return False
    
    # Test imports from duckdb_manager
    try:
        from duckdb_manager import (
            DatabaseTable,
            QueryResult,
            DuckDBManager,
            create_in_memory_db
        )
        print("✅ duckdb_manager imports OK")
    except ImportError as e:
        print(f"❌ duckdb_manager import error: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("AI Data Analysis Agent - Import Test")
    print("=" * 60)
    
    success = True
    
    # Test basic imports
    if not test_imports():
        success = False
    
    # Test app-specific imports
    if not test_app_specific():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - Application should run correctly!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Please check dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main())