from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import os

import duckdb
from openai import OpenAI


@dataclass
class TableInfo:
    name: str
    columns: list[tuple[str, str]]
    row_count: int


def get_table_info(connection: duckdb.DuckDBPyConnection) -> list[TableInfo]:
    tables = connection.execute("SHOW TABLES").fetchall()
    info: list[TableInfo] = []
    for (table_name,) in tables:
        cols = connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        columns = [(c[1], c[2]) for c in cols]
        row_count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        info.append(TableInfo(name=table_name, columns=columns, row_count=row_count))
    return info


def render_schema_context(tables: Iterable[TableInfo]) -> str:
    lines: list[str] = []
    for table in tables:
        lines.append(f"Table: {table.name} (rows: {table.row_count})")
        for col_name, col_type in table.columns:
            lines.append(f"- {col_name}: {col_type}")
        lines.append("")
    return "\n".join(lines).strip()


def build_agent(api_key: str, schema_context: str):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    instructions = [
        "You are a data analyst. Convert the user question into a single DuckDB SQL query.",
        "Only return SQL. Do not return explanations, markdown, or code fences.",
        "Use only the tables and columns provided in the schema context.",
        "Prefer simple, readable SQL. Use LIMIT 200 for large result sets unless the user asks for all rows.",
    ]
    
    class SimpleAgent:
        def __init__(self, client, instructions, context):
            self.client = client
            self.instructions = instructions
            self.context = context
        
        def run(self, question: str):
            system_prompt = "\n".join(self.instructions)
            full_context = f"{self.context}\n\nUser Question: {question}"
            
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_context},
                    ],
                    stream=False,
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling DeepSeek API: {str(e)}"
    
    return SimpleAgent(
        client=client,
        instructions=instructions,
        context=f"Schema context:\n{schema_context}"
    )


def extract_sql(response) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()