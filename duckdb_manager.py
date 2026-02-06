from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd


@dataclass
class DatabaseTable:
    name: str
    columns: list[tuple[str, str]]
    row_count: int
    size_mb: float
    created_at: str


@dataclass
class QueryResult:
    success: bool
    data: pd.DataFrame | None
    error: str | None
    execution_time_ms: float
    row_count: int
    column_names: list[str]
    sql: str


class DuckDBManager:
    def __init__(self, database_path: str = ":memory:", persist: bool = False):
        self.database_path = database_path
        self.persist = persist
        self.connection: duckdb.DuckDBPyConnection | None = None
        self._connect()
    
    def _connect(self) -> None:
        if self.connection is not None:
            self.connection.close()
        
        if self.persist and self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.connection = duckdb.connect(database=self.database_path)
        self._configure_connection()
    
    def _configure_connection(self) -> None:
        if self.connection is None:
            return
        
        self.connection.execute("SET threads TO 4")
        self.connection.execute("SET memory_limit='2GB'")
        self.connection.execute("PRAGMA enable_progress_bar=true")
    
    def load_csv(self, file_path: Path | str, table_name: Optional[str] = None, 
                 delimiter: str = ",", header: bool = True) -> str:
        file_path = Path(file_path)
        if not table_name:
            table_name = self._generate_table_name(file_path.stem)
        
        table_name = self._ensure_unique_table_name(table_name)
        
        query = f"""
        CREATE TABLE {table_name} AS 
        SELECT * FROM read_csv_auto(
            '{file_path.as_posix()}', 
            HEADER={str(header).upper()},
            DELIM='{delimiter}'
        )
        """
        
        self.connection.execute(query)
        return table_name
    
    def load_excel(self, file_path: Path | str, table_name: Optional[str] = None, 
                   sheet_name: str = "0") -> str:
        file_path = Path(file_path)
        if not table_name:
            table_name = self._generate_table_name(file_path.stem)
        
        table_name = self._ensure_unique_table_name(table_name)
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if df.empty:
                raise ValueError(f"Excel文件 '{file_path.name}' 的工作表 '{sheet_name}' 是空的")
            
            if len(df.columns) == 0:
                raise ValueError(f"Excel文件 '{file_path.name}' 的工作表 '{sheet_name}' 没有数据列")
            
            self.connection.register("_temp_df", df)
            self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _temp_df")
            self.connection.unregister("_temp_df")
            
            return table_name
            
        except Exception as e:
            error_msg = f"加载Excel文件失败: {str(e)}"
            
            try:
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
                error_msg += f"\n可用工作表: {', '.join(sheet_names)}"
                
                for sheet in sheet_names:
                    try:
                        test_df = pd.read_excel(file_path, sheet_name=sheet, nrows=5)
                        error_msg += f"\n工作表 '{sheet}': {len(test_df.columns)} 列, {len(test_df)} 行"
                    except Exception as sheet_error:
                        error_msg += f"\n工作表 '{sheet}': 读取失败 - {str(sheet_error)}"
                        
            except Exception as debug_error:
                error_msg += f"\n调试信息获取失败: {str(debug_error)}"
            
            raise ValueError(error_msg)
    
    def load_parquet(self, file_path: Path | str, table_name: Optional[str] = None) -> str:
        file_path = Path(file_path)
        if not table_name:
            table_name = self._generate_table_name(file_path.stem)
        
        table_name = self._ensure_unique_table_name(table_name)
        
        query = f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path.as_posix()}')"
        self.connection.execute(query)
        return table_name
    
    def load_json(self, file_path: Path | str, table_name: Optional[str] = None) -> str:
        file_path = Path(file_path)
        if not table_name:
            table_name = self._generate_table_name(file_path.stem)
        
        table_name = self._ensure_unique_table_name(table_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        self.connection.register("_temp_df", df)
        self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _temp_df")
        self.connection.unregister("_temp_df")
        
        return table_name
    
    def _generate_table_name(self, base_name: str) -> str:
        import re
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", base_name).strip("_")
        if not name:
            name = "table"
        if name[0].isdigit():
            name = f"t_{name}"
        return name.lower()
    
    def _ensure_unique_table_name(self, base_name: str) -> str:
        existing_tables = self.get_table_names()
        if base_name not in existing_tables:
            return base_name
        
        i = 2
        while f"{base_name}_{i}" in existing_tables:
            i += 1
        return f"{base_name}_{i}"
    
    def get_table_names(self) -> list[str]:
        if self.connection is None:
            return []
        
        result = self.connection.execute("SHOW TABLES").fetchall()
        return [row[0] for row in result]
    
    def get_table_info(self, table_name: str) -> DatabaseTable | None:
        if self.connection is None:
            return None
        
        try:
            columns_query = f"PRAGMA table_info('{table_name}')"
            columns_result = self.connection.execute(columns_query).fetchall()
            columns = [(col[1], col[2]) for col in columns_result]
            
            row_count = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            size_query = f"""
            SELECT SUM(block_size * row_group_count) 
            FROM pragma_storage_info('{table_name}')
            """
            try:
                size_bytes = self.connection.execute(size_query).fetchone()[0] or 0
            except:
                size_bytes = 0
            
            size_mb = size_bytes / (1024 * 1024)
            
            return DatabaseTable(
                name=table_name,
                columns=columns,
                row_count=row_count,
                size_mb=size_mb,
                created_at=self._get_creation_time(table_name)
            )
        except Exception:
            return None
    
    def _get_creation_time(self, table_name: str) -> str:
        try:
            import datetime
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except:
            return "Unknown"
    
    def get_all_table_info(self) -> list[DatabaseTable]:
        tables = []
        for table_name in self.get_table_names():
            info = self.get_table_info(table_name)
            if info:
                tables.append(info)
        return tables
    
    def execute_query(self, sql: str, params: Optional[dict] = None) -> QueryResult:
        import time
        
        if self.connection is None:
            return QueryResult(
                success=False,
                data=None,
                error="Database connection not available",
                execution_time_ms=0,
                row_count=0,
                column_names=[],
                sql=sql
            )
        
        start_time = time.time()
        
        try:
            if params:
                result = self.connection.execute(sql, params)
            else:
                result = self.connection.execute(sql)
            
            df = result.fetchdf()
            execution_time = (time.time() - start_time) * 1000
            
            return QueryResult(
                success=True,
                data=df,
                error=None,
                execution_time_ms=execution_time,
                row_count=len(df),
                column_names=list(df.columns) if not df.empty else [],
                sql=sql
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return QueryResult(
                success=False,
                data=None,
                error=str(e),
                execution_time_ms=execution_time,
                row_count=0,
                column_names=[],
                sql=sql
            )
    
    def get_table_sample(self, table_name: str, limit: int = 10) -> pd.DataFrame | None:
        if self.connection is None:
            return None
        
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return self.connection.execute(query).fetchdf()
        except Exception:
            return None
    
    def get_table_statistics(self, table_name: str) -> dict[str, Any]:
        if self.connection is None:
            return {}
        
        try:
            stats = {}
            
            column_types = self.connection.execute(f"DESCRIBE {table_name}").fetchall()
            stats["columns"] = [{"name": col[0], "type": col[1]} for col in column_types]
            
            row_count = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            stats["row_count"] = row_count
            
            numeric_columns = []
            for col in column_types:
                if any(num_type in col[1].lower() for num_type in ["int", "float", "double", "decimal", "real"]):
                    numeric_columns.append(col[0])
            
            if numeric_columns:
                stats["numeric_columns"] = numeric_columns
                
                for col in numeric_columns[:3]:
                    col_stats = self.connection.execute(f"""
                        SELECT 
                            MIN({col}) as min_val,
                            MAX({col}) as max_val,
                            AVG({col}) as avg_val,
                            STDDEV({col}) as std_val,
                            COUNT(DISTINCT {col}) as distinct_count
                        FROM {table_name}
                    """).fetchone()
                    
                    stats[f"{col}_stats"] = {
                        "min": col_stats[0],
                        "max": col_stats[1],
                        "avg": col_stats[2],
                        "std": col_stats[3],
                        "distinct": col_stats[4]
                    }
            
            return stats
        except Exception:
            return {}
    
    def export_to_csv(self, table_name: str, output_path: Path | str) -> bool:
        if self.connection is None:
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            query = f"COPY {table_name} TO '{output_path.as_posix()}' (FORMAT CSV, HEADER TRUE)"
            self.connection.execute(query)
            return True
        except Exception:
            return False
    
    def export_to_parquet(self, table_name: str, output_path: Path | str) -> bool:
        if self.connection is None:
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            query = f"COPY {table_name} TO '{output_path.as_posix()}' (FORMAT PARQUET)"
            self.connection.execute(query)
            return True
        except Exception:
            return False
    
    def create_index(self, table_name: str, column_name: str, index_name: Optional[str] = None) -> bool:
        if self.connection is None:
            return False
        
        if not index_name:
            index_name = f"idx_{table_name}_{column_name}"
        
        try:
            query = f"CREATE INDEX {index_name} ON {table_name} ({column_name})"
            self.connection.execute(query)
            return True
        except Exception:
            return False
    
    def drop_table(self, table_name: str) -> bool:
        if self.connection is None:
            return False
        
        try:
            self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            return True
        except Exception:
            return False
    
    def vacuum(self) -> bool:
        if self.connection is None:
            return False
        
        try:
            self.connection.execute("VACUUM")
            return True
        except Exception:
            return False
    
    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_in_memory_db() -> DuckDBManager:
    return DuckDBManager(database_path=":memory:", persist=False)


def create_persistent_db(db_path: str) -> DuckDBManager:
    return DuckDBManager(database_path=db_path, persist=True)