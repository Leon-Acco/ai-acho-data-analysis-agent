from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent_core import build_agent, extract_sql, get_table_info, render_schema_context
from model_config import (
    AVAILABLE_MODELS,
    get_model_config,
    get_model_env_var,
    get_default_model,
    ModelProvider,
)

load_dotenv()

APP_TITLE = "AI 数据分析 Agent"
UPLOAD_DIR = Path("data_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def normalize_table_name(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
    if not name:
        name = "table"
    if name[0].isdigit():
        name = f"t_{name}"
    return name.lower()


def ensure_unique_table_name(connection: duckdb.DuckDBPyConnection, base_name: str) -> str:
    existing = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if base_name not in existing:
        return base_name
    i = 2
    while f"{base_name}_{i}" in existing:
        i += 1
    return f"{base_name}_{i}"


def load_csv(connection: duckdb.DuckDBPyConnection, file_path: Path) -> str:
    table_base = normalize_table_name(file_path.name)
    table_name = ensure_unique_table_name(connection, table_base)
    connection.execute(
        f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path.as_posix()}', HEADER=True)"
    )
    return table_name


def load_excel(connection: duckdb.DuckDBPyConnection, file_path: Path) -> str:
    table_base = normalize_table_name(file_path.name)
    table_name = ensure_unique_table_name(connection, table_base)
    df = pd.read_excel(file_path)
    connection.register("_tmp_df", df)
    connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _tmp_df")
    connection.unregister("_tmp_df")
    return table_name


def save_upload(upload) -> Path:
    dest = UPLOAD_DIR / upload.name
    dest.write_bytes(upload.getvalue())
    return dest


def init_state() -> None:
    if "connection" not in st.session_state:
        st.session_state.connection = duckdb.connect(database=":memory:")
    if "tables" not in st.session_state:
        st.session_state.tables = []
    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = get_default_model()


def reset_state() -> None:
    if "connection" in st.session_state:
        st.session_state.connection.close()
    st.session_state.clear()
    init_state()


def get_api_key_from_env(model_key: str) -> str:
    """从环境变量获取指定模型的API密钥"""
    model_config = get_model_config(model_key)
    if model_config is None:
        return ""
    
    env_var = get_model_env_var(model_config.provider)
    return os.getenv(env_var, "")


def render_sidebar() -> tuple[str, str, int]:
    """渲染侧边栏设置
    
    Returns:
        tuple: (api_key, model_key, row_limit)
    """
    with st.sidebar:
        st.header("⚙️ 设置")
        
        st.subheader("🤖 模型选择")
        
        model_options = {
            config.display_name: key for key, config in AVAILABLE_MODELS.items()
        }
        
        default_model = st.session_state.get("selected_model", get_default_model())
        default_index = list(model_options.values()).index(default_model) if default_model in model_options.values() else 0
        
        selected_display = st.selectbox(
            "选择模型",
            options=list(model_options.keys()),
            index=default_index,
            help="选择要使用的AI模型"
        )
        
        model_key = model_options.get(selected_display, get_default_model())
        st.session_state.selected_model = model_key
        
        model_config = get_model_config(model_key)
        provider_name = model_config.provider.value.upper() if model_config else "UNKNOWN"
        st.caption(f"提供商: {provider_name}")
        
        st.divider()
        
        st.subheader("🔑 API密钥")
        
        env_api_key = get_api_key_from_env(model_key)
        
        api_key = st.text_input(
            f"{selected_display} API密钥",
            value=env_api_key,
            type="password",
            help=f"请输入 {selected_display} 的API密钥，支持环境变量配置"
        )
        
        if not api_key and env_api_key:
            api_key = env_api_key
            st.success("✅ 已从环境变量加载API密钥")
        elif api_key:
            st.success("✅ 已配置API密钥")
        else:
            st.warning("⚠️ 请配置API密钥以使用AI功能")
        
        st.divider()
        
        st.subheader("📊 查询设置")
        row_limit = st.number_input(
            "默认行数限制",
            min_value=50,
            max_value=5000,
            value=200,
            step=50,
            help="查询返回的最大行数"
        )
        
        st.divider()
        
        if st.button("🔄 重置工作区", use_container_width=True):
            reset_state()
            st.rerun()
        
        st.divider()
        st.subheader("📁 已加载表")
        if st.session_state.tables:
            for table in st.session_state.tables:
                st.write(f"- {table}")
        else:
            st.caption("尚未加载任何表。")
        
        st.divider()
        st.caption("使用 Agno & DuckDB 构建")
    
    return api_key, model_key, row_limit


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"📊 {APP_TITLE}")
    st.caption("支持多模型的数据分析工具 | DeepSeek、智谱、豆包、千问、OpenAI")
    
    init_state()
    
    api_key, model_key, row_limit = render_sidebar()
    
    st.subheader("📤 上传数据")
    uploads = st.file_uploader(
        "上传CSV或Excel文件",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    
    if uploads:
        for upload in uploads:
            upload_key = f"{upload.name}:{upload.size}"
            if upload_key in st.session_state.processed_uploads:
                continue
            dest = save_upload(upload)
            try:
                if dest.suffix.lower() == ".csv":
                    table_name = load_csv(st.session_state.connection, dest)
                else:
                    table_name = load_excel(st.session_state.connection, dest)
                st.session_state.tables.append(table_name)
                st.session_state.processed_uploads.add(upload_key)
                st.success(f"已加载 {upload.name} 到表 `{table_name}`")
            except Exception as exc:
                st.error(f"加载 {upload.name} 失败: {exc}")
    
    schema_context = render_schema_context(get_table_info(st.session_state.connection))
    
    st.markdown("---")
    st.subheader("🔍 关于你的数据提问")
    question = st.text_input(
        "使用自然语言提问",
        placeholder="例如：'按销售额排名前10的客户' 或 '月收入趋势如何？'"
    )
    
    if st.button("🚀 运行分析", type="primary"):
        if not api_key:
            st.error("请在侧边栏配置API密钥。")
        elif not question.strip():
            st.error("请输入问题。")
        elif not st.session_state.tables:
            st.error("请至少上传一个数据文件。")
        else:
            model_config = get_model_config(model_key)
            with st.spinner(f"🤖 正在使用 {model_config.display_name} 生成SQL..."):
                agent = build_agent(
                    api_key=api_key,
                    schema_context=schema_context,
                    model_key=model_key
                )
                response = agent.run(question)
            
            sql = extract_sql(response)
            
            if not sql:
                st.error("AI未返回有效的SQL，请尝试重新表述问题。")
            else:
                try:
                    if "limit" not in sql.lower():
                        sql = f"{sql.rstrip(';')} LIMIT {int(row_limit)}"
                    
                    df = st.session_state.connection.execute(sql).fetchdf()
                    
                    st.session_state.history.append({
                        "question": question,
                        "sql": sql,
                        "rows": len(df),
                        "model": model_config.display_name
                    })
                    
                    st.markdown("**🤖 生成的SQL**")
                    st.code(sql, language="sql")
                    
                    if df.empty:
                        st.info("查询未返回任何数据。")
                    else:
                        st.markdown("**📊 查询结果**")
                        st.dataframe(df, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("行数", f"{len(df):,}")
                        with col2:
                            st.metric("列数", len(df.columns))
                        
                        numeric_cols = df.select_dtypes(include="number").columns.tolist()
                        if numeric_cols and len(df.columns) >= 2:
                            st.markdown("**📈 快速可视化**")
                            x_col = st.selectbox("X轴", df.columns.tolist(), index=0, key="x_axis")
                            y_col = st.selectbox("Y轴", numeric_cols, index=0 if numeric_cols else None, key="y_axis")
                            if x_col and y_col:
                                try:
                                    chart_df = df[[x_col, y_col]].set_index(x_col)
                                    if chart_df.index.nlevels > 1:
                                        chart_df = df[[x_col, y_col]]
                                    st.line_chart(chart_df)
                                except Exception as e:
                                    st.warning(f"无法创建图表: {str(e)}")
                                    st.bar_chart(df[y_col])
                
                except Exception as exc:
                    st.error(f"查询执行失败: {exc}")
    
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 分析历史")
        for item in reversed(st.session_state.history[-10:]):
            model_tag = item.get("model", "")
            st.write(f"**Q:** {item['question']}")
            if model_tag:
                st.caption(f"模型: {model_tag}")
            st.caption(f"SQL: {item['sql']}")
            st.caption(f"结果: {item['rows']} 行")


if __name__ == "__main__":
    main()
