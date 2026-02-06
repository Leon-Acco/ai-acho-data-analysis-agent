from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent_enhanced import (
    build_advanced_agent,
    extract_sql_from_response,
    generate_analysis_summary,
    get_table_info_with_samples,
    render_schema_context,
    suggest_visualizations,
    validate_sql,
)
from duckdb_manager import DuckDBManager, create_in_memory_db

load_dotenv()

APP_TITLE = "AI数据分析助手 (Agno + DeepSeek + DuckDB)"
APP_DESCRIPTION = "上传CSV/Excel文件并使用自然语言提问。通过AI生成SQL实现即时数据分析。"


def init_session_state() -> None:
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = create_in_memory_db()
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
    
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    
    if "show_advanced" not in st.session_state:
        st.session_state.show_advanced = False


def reset_session() -> None:
    if "db_manager" in st.session_state:
        st.session_state.db_manager.close()
    
    st.session_state.clear()
    init_session_state()
    st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.title("⚙️ 设置")
        
        api_key = st.text_input(
            "DeepSeek API密钥",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            type="password",
            help="从 https://platform.deepseek.com/ 获取您的API密钥"
        )
        
        st.session_state.api_key = api_key
        
        st.divider()
        
        st.subheader("📊 查询设置")
        row_limit = st.number_input(
            "默认行数限制",
            min_value=10,
            max_value=10000,
            value=200,
            step=10,
            help="查询返回的最大行数"
        )
        st.session_state.row_limit = row_limit
        
        enable_explanations = st.checkbox(
            "显示SQL解释",
            value=False,
            help="包含生成SQL的解释说明"
        )
        st.session_state.enable_explanations = enable_explanations
        
        st.divider()
        
        st.subheader("📁 已加载表")
        
        if st.session_state.db_manager:
            tables = st.session_state.db_manager.get_all_table_info()
            if tables:
                for table in tables:
                    with st.expander(f"📊 {table.name}"):
                        st.write(f"**行数:** {table.row_count:,}")
                        st.write(f"**大小:** {table.size_mb:.2f} MB")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"预览", key=f"preview_{table.name}"):
                                st.session_state[f"preview_{table.name}"] = True
                        
                        with col2:
                            if st.button(f"统计", key=f"stats_{table.name}"):
                                st.session_state[f"stats_{table.name}"] = True
                        
                        if st.session_state.get(f"preview_{table.name}", False):
                            sample = st.session_state.db_manager.get_table_sample(table.name, limit=5)
                            if sample is not None:
                                st.dataframe(sample, use_container_width=True)
                        
                        if st.session_state.get(f"stats_{table.name}", False):
                            stats = st.session_state.db_manager.get_table_statistics(table.name)
                            if stats:
                                st.json(stats, expanded=False)
            else:
                st.info("尚未加载任何表。请在上方上传文件。")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重置工作区", use_container_width=True):
                reset_session()
        
        with col2:
            if st.button("💾 导出会话", use_container_width=True):
                st.session_state.show_export = True
        
        st.divider()
        st.caption("使用 Agno、DeepSeek & DuckDB 构建")


def render_file_upload() -> None:
    st.header("📤 上传数据文件")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        file_types = st.multiselect(
            "文件格式",
            options=["CSV", "Excel", "Parquet", "JSON"],
            default=["CSV", "Excel"],
            help="选择支持的文件格式"
        )
    
    with col2:
        delimiter = st.selectbox(
            "CSV分隔符",
            options=[",", ";", "\t", "|"],
            index=0,
            help="CSV文件的列分隔符"
        )
    
    with col3:
        sheet_option = st.selectbox(
            "Excel工作表",
            options=["第一个工作表", "所有工作表", "指定名称"],
            index=0,
            help="Excel工作表加载选项"
        )
    
    uploaded_files = st.file_uploader(
        "选择文件",
        type=get_file_extensions(file_types),
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if file_key in st.session_state.uploaded_files:
                st.info(f"文件 '{uploaded_file.name}' 已加载")
                continue
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)
            
            try:
                file_suffix = Path(uploaded_file.name).suffix.lower()
                
                if file_suffix == ".csv":
                    table_name = st.session_state.db_manager.load_csv(
                        tmp_path, delimiter=delimiter
                    )
                elif file_suffix in [".xlsx", ".xls"]:
                    if sheet_option == "第一个工作表":
                        table_name = st.session_state.db_manager.load_excel(tmp_path, sheet_name=0)
                    elif sheet_option == "所有工作表":
                        try:
                            xls = pd.ExcelFile(tmp_path)
                            sheet_names = xls.sheet_names
                            tables_loaded = []
                            for sheet in sheet_names:
                                try:
                                    table_name = st.session_state.db_manager.load_excel(
                                        tmp_path, 
                                        sheet_name=sheet,
                                        table_name=f"{Path(uploaded_file.name).stem}_{sheet}"
                                    )
                                    tables_loaded.append(table_name)
                                except Exception as sheet_error:
                                    st.warning(f"工作表 '{sheet}' 加载失败: {str(sheet_error)}")
                            
                            if tables_loaded:
                                table_name = tables_loaded[0]
                                st.success(f"✅ 已加载 {len(tables_loaded)} 个工作表: {', '.join(tables_loaded)}")
                            else:
                                raise ValueError("所有工作表加载都失败了")
                        except Exception as e:
                            raise ValueError(f"加载所有工作表失败: {str(e)}")
                    else:
                        sheet_name_input = st.text_input("请输入工作表名称", key=f"sheet_name_{uploaded_file.name}")
                        if sheet_name_input:
                            table_name = st.session_state.db_manager.load_excel(tmp_path, sheet_name=sheet_name_input)
                        else:
                            raise ValueError("请指定工作表名称")
                elif file_suffix == ".parquet":
                    table_name = st.session_state.db_manager.load_parquet(tmp_path)
                elif file_suffix == ".json":
                    table_name = st.session_state.db_manager.load_json(tmp_path)
                else:
                    st.error(f"不支持的文件格式: {file_suffix}")
                    continue
                
                st.session_state.uploaded_files[file_key] = {
                    "name": uploaded_file.name,
                    "table": table_name,
                    "size": uploaded_file.size
                }
                
                st.success(f"✅ 已加载 '{uploaded_file.name}' 为表 `{table_name}`")
                
                table_info = st.session_state.db_manager.get_table_info(table_name)
                if table_info:
                    with st.expander(f"📋 Schema: {table_name}", expanded=False):
                        st.write(f"**列数:** {len(table_info.columns)}")
                        for col_name, col_type in table_info.columns[:10]:
                            st.code(f"{col_name}: {col_type}")
                        if len(table_info.columns) > 10:
                            st.caption(f"... 还有 {len(table_info.columns) - 10} 列")
                
            except Exception as e:
                st.error(f"加载文件 '{uploaded_file.name}' 失败: {str(e)}")
                with st.expander("🛠️ 详细错误信息"):
                    st.text(str(e))
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if uploaded_files:
            progress_bar.empty()


def get_file_extensions(file_types: list[str]) -> list[str]:
    extensions = []
    if "CSV" in file_types:
        extensions.extend([".csv", ".tsv"])
    if "Excel" in file_types:
        extensions.extend([".xlsx", ".xls"])
    if "Parquet" in file_types:
        extensions.extend([".parquet"])
    if "JSON" in file_types:
        extensions.extend([".json"])
    return extensions


def render_query_interface() -> None:
    st.header("🔍 关于你的数据提问")
    
    if not st.session_state.db_manager.get_table_names():
        st.info("📁 请先上传数据文件以开始分析")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_area(
            "使用自然语言提问",
            placeholder="例如：'显示前10名总购买额最高的客户' 或 '每月的平均销售额是多少？'",
            height=100,
            key="question_input"
        )
    
    with col2:
        st.write("###")
        if st.button("🚀 分析", type="primary", use_container_width=True):
            st.session_state.current_query = question
        
        if st.button("💡 示例问题", use_container_width=True):
            st.session_state.show_examples = True
    
    if st.session_state.get("show_examples", False):
        with st.expander("📝 示例问题", expanded=True):
            examples = [
                "销售额排名前5的产品是什么？",
                "显示月收入趋势",
                "按地区比较销售额",
                "平均客户年龄是多少？",
                "在数据中查找重复记录",
                "计算年增长率",
                "显示订单值的分布情况",
                "一周中哪天的销售额最高？",
                "查找变量之间的相关性",
                "基于历史数据预测下月销售额"
            ]
            
            for example in examples:
                if st.button(example, use_container_width=True, key=f"example_{example}"):
                    st.session_state.current_query = example
                    st.rerun()
    
    if st.session_state.current_query and st.session_state.api_key:
        process_query(st.session_state.current_query)


def process_query(question: str) -> None:
    with st.spinner("🤖 生成SQL查询..."):
        tables = get_table_info_with_samples(st.session_state.db_manager.connection)
        schema_context = render_schema_context(tables, include_samples=True)
        
        agent = build_advanced_agent(
            api_key=st.session_state.api_key,
            schema_context=schema_context,
            enable_explanations=st.session_state.enable_explanations
        )
        
        response = agent.run(question)
        sql = extract_sql_from_response(response, st.session_state.enable_explanations)
    
    if not sql:
        st.error("AI代理未生成有效的SQL。请尝试重新表述您的问题。")
        return
    
    st.subheader("📄 生成的SQL")
    st.code(sql, language="sql")
    
    with st.spinner("🔍 验证并执行查询..."):
        is_valid, validation_msg = validate_sql(sql, st.session_state.db_manager.connection)
        
        if not is_valid:
            st.error(f"SQL验证失败: {validation_msg}")
            return
        
        st.success(f"✅ SQL验证通过: {validation_msg}")
        
        if "limit" not in sql.lower():
            sql = f"{sql.rstrip(';')} LIMIT {st.session_state.row_limit}"
        
        result = st.session_state.db_manager.execute_query(sql)
    
    if result.success:
        display_query_results(result, question, sql)
    else:
        st.error(f"查询执行失败: {result.error}")
        
        with st.expander("🛠️ 调试信息"):
            st.write("**Schema上下文:**")
            st.text(schema_context[:500] + "..." if len(schema_context) > 500 else schema_context)
            
            st.write("**Agent响应:**")
            st.text(str(response)[:500] + "..." if len(str(response)) > 500 else str(response))


def display_query_results(result, question: str, sql: str) -> None:
    df = result.data
    
    st.subheader("📊 结果")
    
    tabs = st.tabs(["📋 数据", "📈 摘要", "📊 可视化", "💾 导出"])
    
    with tabs[0]:
        st.dataframe(df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("行数", f"{len(df):,}")
        with col2:
            st.metric("列数", len(df.columns))
        with col3:
            st.metric("执行时间", f"{result.execution_time_ms:.1f} 毫秒")
    
    with tabs[1]:
        summary = generate_analysis_summary(df)
        
        st.subheader("统计摘要")
        
        if "numeric_columns" in summary:
            st.write("**数值列:**")
            for col in summary["numeric_columns"][:5]:
                if f"{col}_stats" in summary:
                    stats = summary[f"{col}_stats"]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("最小值", f"{stats.get('min', 0):.2f}")
                    with col2:
                        st.metric("最大值", f"{stats.get('max', 0):.2f}")
                    with col3:
                        st.metric("平均值", f"{stats.get('mean', 0):.2f}")
                    with col4:
                        st.metric("标准差", f"{stats.get('std', 0):.2f}")
        
        if "string_columns" in summary:
            st.write("**字符串列:**")
            for col in summary["string_columns"][:3]:
                if f"{col}_info" in summary:
                    info = summary[f"{col}_info"]
                    st.write(f"**{col}:** {info.get('unique_count', 0)} 个唯一值")
                    if info.get("sample_values"):
                        st.write(f"示例: {', '.join(map(str, info['sample_values'][:3]))}")
        
        st.divider()
        st.write("**数据类型:**")
        for col, dtype in summary.get("data_types", {}).items():
            st.code(f"{col}: {dtype}")
    
    with tabs[2]:
        if not df.empty:
            suggestions = suggest_visualizations(df)
            
            st.write("**建议的可视化:**")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
            
            st.divider()
            
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            categorical_cols = df.select_dtypes(include="object").columns.tolist()
            date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
            
            if numeric_cols and (len(categorical_cols) >= 1 or len(date_cols) >= 1):
                st.subheader("创建可视化")
                
                viz_type = st.selectbox(
                    "图表类型",
                    options=["折线图", "柱状图", "散点图", "面积图", "直方图", "箱线图"],
                    index=0
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox(
                        "X轴",
                        options=df.columns.tolist(),
                        index=0
                    )
                
                with col2:
                    if viz_type in ["直方图", "箱线图"]:
                        y_col = st.selectbox(
                            "数值列",
                            options=numeric_cols,
                            index=0 if numeric_cols else None
                        )
                    else:
                        y_col = st.selectbox(
                            "Y轴",
                            options=numeric_cols,
                            index=0 if numeric_cols else None
                        )
                
                if viz_type == "折线图" and x_col and y_col:
                    chart_data = df[[x_col, y_col]].set_index(x_col)
                    st.line_chart(chart_data)
                elif viz_type == "柱状图" and x_col and y_col:
                    chart_data = df[[x_col, y_col]].set_index(x_col)
                    st.bar_chart(chart_data)
                elif viz_type == "散点图" and x_col and y_col:
                    st.scatter_chart(df, x=x_col, y=y_col)
                elif viz_type == "面积图" and x_col and y_col:
                    chart_data = df[[x_col, y_col]].set_index(x_col)
                    st.area_chart(chart_data)
                elif viz_type == "直方图" and y_col:
                    st.bar_chart(df[y_col].value_counts().sort_index())
                elif viz_type == "箱线图" and y_col:
                    if categorical_cols:
                        group_col = st.selectbox("分组依据", categorical_cols)
                        groups = df[group_col].unique()
                        for group in groups[:5]:
                            group_data = df[df[group_col] == group][y_col]
                            st.write(f"**{group}:** 最小值={group_data.min():.2f}, 最大值={group_data.max():.2f}, 平均值={group_data.mean():.2f}")
                    else:
                        st.write(f"**{y_col}的统计信息:**")
                        st.write(f"最小值={df[y_col].min():.2f}, 最大值={df[y_col].max():.2f}, 平均值={df[y_col].mean():.2f}")
            
            elif numeric_cols and len(numeric_cols) >= 2:
                st.subheader("相关矩阵")
                corr_matrix = df[numeric_cols].corr()
                st.dataframe(corr_matrix.round(3))
    
    with tabs[3]:
        st.subheader("导出选项")
        
        export_format = st.selectbox(
            "导出格式",
            options=["CSV", "Excel", "JSON", "Parquet"],
            index=0
        )
        
        export_filename = st.text_input(
            "文件名",
            value=f"analysis_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            help="导出文件的名称（不含扩展名）"
        )
        
        if st.button("📥 导出数据", type="primary"):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / f"{export_filename}.{export_format.lower()}"
                
                try:
                    if export_format == "CSV":
                        df.to_csv(tmp_path, index=False)
                    elif export_format == "Excel":
                        df.to_excel(tmp_path, index=False)
                    elif export_format == "JSON":
                        df.to_json(tmp_path, orient="records", indent=2)
                    elif export_format == "Parquet":
                        df.to_parquet(tmp_path, index=False)
                    
                    with open(tmp_path, "rb") as f:
                        st.download_button(
                            label=f"下载 {export_format} 文件",
                            data=f,
                            file_name=f"{export_filename}.{export_format.lower()}",
                            mime={
                                "CSV": "text/csv",
                                "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                "JSON": "application/json",
                                "Parquet": "application/octet-stream"
                            }[export_format]
                        )
                    
                    st.success(f"数据已成功导出为 {export_format} 格式")
                except Exception as e:
                    st.error(f"导出失败: {str(e)}")
        
        st.divider()
        
        st.subheader("保存分析到历史记录")
        if st.button("💾 保存到历史记录"):
            analysis_entry = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "question": question,
                "sql": sql,
                "row_count": len(df),
                "execution_time": result.execution_time_ms,
                "columns": list(df.columns)
            }
            
            st.session_state.analysis_history.append(analysis_entry)
            st.success("分析已保存到历史记录！")
    
    st.session_state.analysis_history.append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "question": question,
        "sql": sql,
        "row_count": len(df),
        "execution_time": result.execution_time_ms
    })


def render_history() -> None:
    if not st.session_state.analysis_history:
        return
    
    st.header("📜 分析历史")
    
    for i, entry in enumerate(reversed(st.session_state.analysis_history[-10:])):
        with st.expander(f"查询 {len(st.session_state.analysis_history) - i}: {entry['question'][:50]}..."):
            st.write(f"**时间:** {entry['timestamp']}")
            st.write(f"**问题:** {entry['question']}")
            st.code(entry['sql'], language="sql")
            st.write(f"**结果:** {entry.get('row_count', 0)} 行, {entry.get('execution_time', 0):.1f} 毫秒")
            
            if st.button(f"重新运行查询", key=f"rerun_{i}"):
                st.session_state.current_query = entry['question']
                st.rerun()


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    
    init_session_state()
    render_sidebar()
    
    render_file_upload()
    render_query_interface()
    render_history()
    
    st.divider()
    st.caption("✨ 由 [Agno](https://github.com/agno-agi/agno)、[DeepSeek](https://platform.deepseek.com/)、[DuckDB](https://duckdb.org/) 和 [Streamlit](https://streamlit.io/) 提供支持")


if __name__ == "__main__":
    main()