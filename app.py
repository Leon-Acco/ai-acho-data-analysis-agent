from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent_core import build_agent, extract_sql, get_table_info, render_schema_context

load_dotenv()

APP_TITLE = "Agno + DeepSeek 数据分析助手"
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


def reset_state() -> None:
    if "connection" in st.session_state:
        st.session_state.connection.close()
    st.session_state.clear()
    init_state()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    init_state()

    with st.sidebar:
        st.header("设置")
        api_key = st.text_input("DeepSeek API密钥", value=os.getenv("DEEPSEEK_API_KEY", ""), type="password")
        row_limit = st.number_input("默认行数限制", min_value=50, max_value=5000, value=200, step=50)
        if st.button("重置工作区"):
            reset_state()
            st.rerun()

        st.markdown("---")
        st.subheader("已加载表")
        if st.session_state.tables:
            for table in st.session_state.tables:
                st.write(f"- {table}")
        else:
            st.caption("尚未加载任何表。")

    st.subheader("上传数据")
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
                st.success(f"Loaded {upload.name} into table `{table_name}`")
            except Exception as exc:
                st.error(f"Failed to load {upload.name}: {exc}")

    schema_context = render_schema_context(get_table_info(st.session_state.connection))

    st.markdown("---")
    st.subheader("提问")
    question = st.text_input("关于你的数据提问")

    if st.button("运行分析", type="primary"):
        if not api_key:
            st.error("请在侧边栏提供DeepSeek API密钥。")
        elif not question.strip():
            st.error("请输入问题。")
        elif not st.session_state.tables:
            st.error("请至少上传一个数据文件。")
        else:
            agent = build_agent(api_key=api_key, schema_context=schema_context)
            response = agent.run(question)
            sql = extract_sql(response)

            if not sql:
                st.error("代理未返回SQL。请尝试不同的问题。")
            else:
                try:
                    if "limit" not in sql.lower():
                        sql = f"{sql.rstrip(';')} LIMIT {int(row_limit)}"

                    df = st.session_state.connection.execute(sql).fetchdf()

                    st.session_state.history.append({"question": question, "sql": sql, "rows": len(df)})

                    st.markdown("**生成的SQL**")
                    st.code(sql, language="sql")

                    if df.empty:
                        st.info("Query returned no rows.")
                    else:
                        st.markdown("**结果**")
                        st.dataframe(df, use_container_width=True)

                        st.markdown("**快速摘要**")
                        if df.shape == (1, 1):
                            st.write(f"答案: {df.iloc[0, 0]}")
                        elif df.shape[0] == 1:
                            st.write(df.iloc[0].to_dict())
                        else:
                            st.write(f"返回 {len(df)} 行和 {len(df.columns)} 列。")

                        numeric_cols = df.select_dtypes(include="number").columns.tolist()
                        if numeric_cols and len(df.columns) >= 2:
                            st.markdown("**可视化**")
                            x_col = st.selectbox("X轴", df.columns.tolist(), index=0)
                            y_col = st.selectbox("Y轴", numeric_cols, index=0)
                            chart_df = df[[x_col, y_col]].set_index(x_col)
                            st.line_chart(chart_df)

                except Exception as exc:
                    st.error(f"Query failed: {exc}")

    if st.session_state.history:
        st.markdown("---")
        st.subheader("History")
        for item in reversed(st.session_state.history[-10:]):
            st.write(f"Q: {item['question']}")
            st.caption(f"SQL: {item['sql']}")
            st.caption(f"Rows: {item['rows']}")


if __name__ == "__main__":
    main()
