from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Any, Iterable

import duckdb
from openai import OpenAI

from model_config import (
    get_model_config,
    ModelProvider,
    AVAILABLE_MODELS,
)


@dataclass
class TableInfo:
    name: str
    columns: list[tuple[str, str]]
    row_count: int
    sample_data: list[tuple] | None = None


@dataclass
class AnalysisResult:
    sql: str
    dataframe: Any
    summary: dict[str, Any]
    visualization_suggestions: list[str]


def get_table_info_with_samples(connection: duckdb.DuckDBPyConnection, sample_limit: int = 5) -> list[TableInfo]:
    tables = connection.execute("SHOW TABLES").fetchall()
    info: list[TableInfo] = []
    for (table_name,) in tables:
        cols = connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        columns = [(c[1], c[2]) for c in cols]
        row_count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        sample_data = None
        if row_count > 0:
            sample_data = connection.execute(
                f"SELECT * FROM {table_name} LIMIT {sample_limit}"
            ).fetchall()
        
        info.append(TableInfo(
            name=table_name, 
            columns=columns, 
            row_count=row_count,
            sample_data=sample_data
        ))
    return info


def render_schema_context(tables: Iterable[TableInfo], include_samples: bool = True) -> str:
    lines: list[str] = []
    for table in tables:
        lines.append(f"Table: {table.name} (rows: {table.row_count})")
        for col_name, col_type in table.columns:
            lines.append(f"- {col_name}: {col_type}")
        
        if include_samples and table.sample_data and table.row_count > 0:
            lines.append("  Sample data:")
            for i, row in enumerate(table.sample_data[:3]):
                lines.append(f"    Row {i+1}: {row}")
        lines.append("")
    return "\n".join(lines).strip()


def build_advanced_agent(
    api_key: str,
    schema_context: str,
    model_key: str = "deepseek-chat",
    enable_explanations: bool = False,
):
    """
    构建增强版AI Agent

    Args:
        api_key: API密钥
        schema_context: 数据库Schema上下文
        model_key: 模型标识符
        enable_explanations: 是否允许返回解释性内容

    Returns:
        Agent实例
    """
    model_config = get_model_config(model_key)
    
    if model_config is None:
        raise ValueError(f"未知的模型: {model_key}，可选模型: {list(AVAILABLE_MODELS.keys())}")
    
    client = OpenAI(
        api_key=api_key,
        base_url=model_config.api_base,
    )
    
    instructions = [
        "You are an expert data analyst specializing in DuckDB SQL.",
        "Convert the user's natural language question into a single, optimized DuckDB SQL query.",
        "Use only the tables and columns provided in the schema context.",
        "For complex questions, consider using CTEs, window functions, and proper aggregations.",
        "Prefer readable SQL with clear column aliases.",
        "Include appropriate LIMIT clauses for large result sets.",
    ]
    
    if not enable_explanations:
        instructions.append("Only return SQL. Do not include explanations, markdown, or code fences.")
    
    context = f"""Database Schema:
{schema_context}

Important Notes:
- Use DuckDB SQL syntax
- For date/time operations, use DATE_TRUNC, EXTRACT functions
- For string operations, use CONCAT, SUBSTRING, UPPER, LOWER
- For aggregations, use SUM, AVG, MIN, MAX, COUNT, COUNT_DISTINCT
- For window functions, use ROW_NUMBER(), RANK(), SUM() OVER()
- Use CAST when needed for type conversions
"""
    
    class SimpleAgent:
        def __init__(self, client, instructions, context, model_config, enable_explanations):
            self.client = client
            self.instructions = instructions
            self.context = context
            self.model_config = model_config
            self.enable_explanations = enable_explanations
        
        def run(self, question: str):
            system_prompt = "\n".join(self.instructions)
            full_context = f"{self.context}\n\nUser Question: {question}"
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_context},
                    ],
                    stream=False,
                    temperature=0.1,
                    max_tokens=1000,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling {self.model_config.display_name} API: {str(e)}"
    
    return SimpleAgent(
        client=client,
        instructions=instructions,
        context=context,
        model_config=model_config,
        enable_explanations=enable_explanations
    )


def validate_sql(sql: str, connection: duckdb.DuckDBPyConnection) -> tuple[bool, str]:
    """验证SQL语法并检查潜在问题"""
    if not sql:
        return False, "Empty SQL"
    
    sql_lower = sql.lower()
    
    dangerous_patterns = [
        (r"drop\s+table", "DROP TABLE operations are not allowed"),
        (r"delete\s+from", "DELETE operations are not allowed"),
        (r"update\s+\w+\s+set", "UPDATE operations are not allowed"),
        (r"create\s+table", "CREATE TABLE operations are not allowed"),
        (r"alter\s+table", "ALTER TABLE operations are not allowed"),
        (r"insert\s+into", "INSERT operations are not allowed"),
        (r"truncate\s+table", "TRUNCATE operations are not allowed"),
    ]
    
    for pattern, message in dangerous_patterns:
        if re.search(pattern, sql_lower):
            return False, f"Security restriction: {message}"
    
    try:
        test_sql = f"EXPLAIN {sql}"
        connection.execute(test_sql).fetchall()
        return True, "SQL is valid"
    except Exception as e:
        return False, f"SQL validation error: {e}"


def extract_sql_from_response(response, allow_explanations: bool = False) -> str:
    """从Agent响应中提取SQL"""
    if response is None:
        return ""
    
    content = ""
    if hasattr(response, "content"):
        content = str(response.content)
    else:
        content = str(response)
    
    content = content.strip()
    
    if allow_explanations:
        sql_match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        sql_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
    
    return content


def generate_analysis_summary(df) -> dict[str, Any]:
    """生成查询结果的综合摘要"""
    if df.empty:
        return {"message": "Query returned no results"}
    
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        summary["numeric_columns"] = numeric_cols
        for col in numeric_cols[:5]:
            summary[f"{col}_stats"] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
            }
    
    string_cols = df.select_dtypes(include="object").columns.tolist()
    if string_cols:
        summary["string_columns"] = string_cols
        for col in string_cols[:3]:
            unique_values = df[col].nunique()
            sample_values = df[col].dropna().unique()[:5].tolist()
            summary[f"{col}_info"] = {
                "unique_count": unique_values,
                "sample_values": sample_values,
            }
    
    return summary


def suggest_visualizations(df) -> list[str]:
    """根据数据特征推荐合适的可视化方式"""
    suggestions = []
    
    if df.empty:
        return ["No data to visualize"]
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
    
    if len(numeric_cols) >= 2:
        suggestions.append("Scatter plot: Compare two numeric variables")
        suggestions.append("Line chart: Show trends over time (if date column exists)")
    
    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        suggestions.append("Bar chart: Compare numeric values across categories")
        suggestions.append("Box plot: Show distribution across categories")
    
    if len(numeric_cols) >= 1:
        suggestions.append("Histogram: Show distribution of a single numeric variable")
    
    if len(categorical_cols) >= 1:
        suggestions.append("Pie chart: Show proportions of categories")
    
    if date_cols and len(numeric_cols) >= 1:
        suggestions.append("Time series: Show trends over time")
    
    if len(df.columns) >= 3 and len(numeric_cols) >= 2:
        suggestions.append("Heatmap: Show correlations between variables")
    
    return suggestions[:5]
