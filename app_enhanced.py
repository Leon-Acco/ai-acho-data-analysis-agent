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
from model_config import (
    AVAILABLE_MODELS,
    get_model_config,
    get_model_env_var,
    get_default_model,
    ModelProvider,
)

load_dotenv()

APP_TITLE = "AI数据分析助手 (Agno + 多模型 + DuckDB)"
APP_DESCRIPTION = "支持多模型的数据分析工具 | DeepSeek、智谱、豆包、千问、OpenAI"


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
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = get_default_model()
    
    if "api_keys_configured" not in st.session_state:
        st.session_state.api_keys_configured = {}


def reset_session() -> None:
    if "db_manager" in st.session_state:
        st.session_state.db_manager.close()
    
    st.session_state.clear()
    init_session_state()
    st.rerun()


def get_api_key_from_env(model_key: str) -> str:
    """从环境变量获取指定模型的API密钥"""
    model_config = get_model_config(model_key)
    if model_config is None:
        return ""
    
    env_var = get_model_env_var(model_config.provider)
    return os.getenv(env_var, "")


def render_sidebar() -> dict:
    """渲染侧边栏设置
    
    Returns:
        dict: 包含 api_key, model_key, row_limit 等配置
    """
    config = {}
    
    with st.sidebar:
        st.title("⚙️ 设置")
        
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
        config["model_key"] = model_key
        
        model_config = get_model_config(model_key)
        provider_name = model_config.provider.value.upper() if model_config else "UNKNOWN"
        st.caption(f"📡 提供商: {provider_name} | 🎯 模型: {model_config.model_name if model_config else 'unknown'}")
        
        st.divider()
        
        st.subheader("🔑 API密钥配置")
        
        env_api_key = get_api_key_from_env(model_key)
        
        page_api_key = st.text_input(
            f"{selected_display} API密钥",
            value=env_api_key,
            type="password",
            key=f"api_key_{model_key}",
            help=f"请输入 {selected_display} 的API密钥，支持环境变量配置"
        )
        
        if not page_api_key and env_api_key:
            api_key = env_api_key
            st.success("✅ 已从环境变量加载API密钥")
        elif page_api_key:
            api_key = page_api_key
            st.success("✅ 已配置API密钥")
        else:
            api_key = ""
            st.warning("⚠️ 请配置API密钥以使用AI功能")
        
        config["api_key"] = api_key
        
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
        config["row_limit"] = row_limit
        
        enable_explanations = st.checkbox(
            "显示SQL解释",
            value=False,
            help="包含生成SQL的解释说明"
        )
        config["enable_explanations"] = enable_explanations
        
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
        st.caption("🤗 使用 Agno、多模型 & DuckDB 构建")
    
    return config


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
    
    if st.session_state.current_query:
        process_query(st.session_state.current_query)


def process_query(question: str) -> None:
    config = st.session_state.get("sidebar_config", {})
    api_key = config.get("api_key", "")
    model_key = config.get("model_key", get_default_model())
    enable_explanations = config.get("enable_explanations", False)
    
    if not api_key:
        st.error("请在侧边栏配置API密钥以使用AI功能")
        return
    
    model_config = get_model_config(model_key)
    if not model_config:
        st.error(f"未知的模型配置: {model_key}")
        return
    
    with st.spinner(f"🤖 正在使用 {model_config.display_name} 生成SQL查询..."):
        tables = get_table_info_with_samples(st.session_state.db_manager.connection)
        schema_context = render_schema_context(tables, include_samples=True)
        
        agent = build_advanced_agent(
            api_key=api_key,
            schema_context=schema_context,
            model_key=model_key,
            enable_explanations=enable_explanations
        )
        
        response = agent.run(question)
        sql = extract_sql_from_response(response, enable_explanations)
    
    if not sql:
        st.error("AI代理未生成有效的SQL。请尝试重新表述您的问题。")
        return
    
    st.subheader("📄 生成的SQL")
    st.code(sql, language="sql")
    
    if model_config:
        st.caption(f"🤖 模型: {model_config.display_name} | 提供商: {model_config.provider.value.upper()}")
    
    with st.spinner("🔍 验证并执行查询..."):
        is_valid, validation_msg = validate_sql(sql, st.session_state.db_manager.connection)
        
        if not is_valid:
            st.error(f"SQL验证失败: {validation_msg}")
            return
        
        st.success(f"✅ SQL验证通过: {validation_msg}")
        
        if "limit" not in sql.lower():
            sql = f"{sql.rstrip(';')} LIMIT {config.get('row_limit', 200)}"
        
        result = st.session_state.db_manager.execute_query(sql)
    
    if result.success:
        display_query_results(result, question, sql, model_config)
    else:
        st.error(f"查询执行失败: {result.error}")
        
        with st.expander("🛠️ 调试信息"):
            st.write("**Schema上下文:**")
            st.text(schema_context[:500] + "..." if len(schema_context) > 500 else schema_context)
            
            st.write("**Agent响应:**")
            st.text(str(response)[:500] + "..." if len(str(response)) > 500 else str(response))


def display_query_results(result, question: str, sql: str, model_config=None) -> None:
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
                    try:
                        chart_data = df[[x_col, y_col]].set_index(x_col)
                        if chart_data.index.nlevels > 1:
                            st.warning("X轴包含多级索引，使用原始数据绘图")
                            chart_data = df[[x_col, y_col]]
                        st.line_chart(chart_data)
                    except Exception as e:
                        st.warning(f"无法创建折线图: {str(e)}")
                        st.bar_chart(df[y_col])
                elif viz_type == "柱状图" and x_col and y_col:
                    try:
                        chart_data = df[[x_col, y_col]].set_index(x_col)
                        if chart_data.index.nlevels > 1:
                            st.warning("X轴包含多级索引，使用原始数据绘图")
                            chart_data = df[[x_col, y_col]]
                        st.bar_chart(chart_data)
                    except Exception as e:
                        st.warning(f"无法创建柱状图: {str(e)}")
                        st.bar_chart(df[y_col])
                elif viz_type == "散点图" and x_col and y_col:
                    try:
                        st.scatter_chart(df, x=x_col, y=y_col)
                    except Exception as e:
                        st.warning(f"无法创建散点图: {str(e)}")
                elif viz_type == "面积图" and x_col and y_col:
                    try:
                        chart_data = df[[x_col, y_col]].set_index(x_col)
                        if chart_data.index.nlevels > 1:
                            st.warning("X轴包含多级索引，使用原始数据绘图")
                            chart_data = df[[x_col, y_col]]
                        st.area_chart(chart_data)
                    except Exception as e:
                        st.warning(f"无法创建面积图: {str(e)}")
                        st.bar_chart(df[y_col])
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
            help="导出的文件名（不含扩展名）"
        )
        
        if st.button("💾 开始导出", use_container_width=True):
            try:
                if export_format == "CSV":
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 下载 CSV",
                        data=csv,
                        file_name=f"{export_filename}.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    excel_buffer = pd.ExcelWriter(
                        pd.io.excel.ExcelWriter(
                            pd.io.common.BytesIO(),
                            engine="openpyxl"
                        ),
                        engine="openpyxl"
                    )
                    df.to_excel(excel_buffer, index=False, sheet_name="Analysis Results")
                    excel_buffer.close()
                    excel_data = excel_buffer.book.book.getvalue()
                    st.download_button(
                        label="📥 下载 Excel",
                        data=excel_data,
                        file_name=f"{export_filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif export_format == "JSON":
                    json_data = df.to_json(orient="records", force_ascii=False).encode("utf-8")
                    st.download_button(
                        label="📥 下载 JSON",
                        data=json_data,
                        file_name=f"{export_filename}.json",
                        mime="application/json"
                    )
                elif export_format == "Parquet":
                    parquet_buffer = df.to_parquet()
                    st.download_button(
                        label="📥 下载 Parquet",
                        data=parquet_buffer,
                        file_name=f"{export_filename}.parquet",
                        mime="application/octet-stream"
                    )
                
                st.success(f"✅ 已准备 {export_format} 格式导出")
            except Exception as e:
                st.error(f"导出失败: {str(e)}")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        page_icon="📊"
    )
    
    st.title(f"📊 {APP_TITLE}")
    st.caption(APP_DESCRIPTION)
    
    init_session_state()
    
    config = render_sidebar()
    st.session_state.sidebar_config = config
    
    render_file_upload()
    render_query_interface()


if __name__ == "__main__":
    main()
