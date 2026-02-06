import sys
sys.path.insert(0, '.')

print('Testing model_config imports...')
from model_config import (
    AVAILABLE_MODELS,
    get_model_config,
    ModelProvider,
    get_model_env_var,
    get_default_model,
)
print(f'Available models: {len(AVAILABLE_MODELS)}')
for key, config in AVAILABLE_MODELS.items():
    print(f'  - {key}: {config.display_name}')

print('\nTesting agent_core imports...')
from agent_core import build_agent, extract_sql, get_table_info, render_schema_context
print('agent_core imported successfully')

print('\nTesting agent_enhanced imports...')
from agent_enhanced import (
    build_advanced_agent,
    validate_sql,
    extract_sql_from_response,
    generate_analysis_summary,
    suggest_visualizations,
)
print('agent_enhanced imported successfully')

print('\nAll imports successful!')
