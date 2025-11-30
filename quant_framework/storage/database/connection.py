"""
数据库连接管理器

提供数据库连接池和会话管理功能，支持多种数据库类型。
"""

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Union

from sqlalchemy import create_engine, event, exc, orm
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from quant_framework.utils.config_loader import get_config

logger = logging.getLogger(__name__)

# 声明基类，所有ORM模型都应继承自此
Base = declarative_base()


class DatabaseConnectionManager:
    """数据库连接管理器
    
    提供数据库连接池和会话管理功能，支持多种数据库类型。
    """
    
    def __init__(self):
        self._engines: Dict[str, Engine] = {}
        self._session_makers: Dict[str, sessionmaker] = {}
        self._lock = threading.Lock()
        self._initialized = False
    
    def initialize(self) -> None:
        """初始化数据库连接"""
        with self._lock:
            if self._initialized:
                return
            
            config = get_config()
            db_config = config.get("database", {})
            
            # 初始化默认数据库连接
            default_url = db_config.get("url", "sqlite:///quant_framework.db")
            self.add_connection("default", default_url)
            
            # 初始化其他数据库连接
            connections = db_config.get("connections", {})
            for name, url in connections.items():
                self.add_connection(name, url)
            
            self._initialized = True
            logger.info(f"数据库连接管理器初始化完成，共{len(self._engines)}个连接")
    
    def add_connection(self, name: str, url: str, **kwargs) -> None:
        """添加数据库连接
        
        Args:
            name: 连接名称
            url: 数据库URL
            **kwargs: 其他连接参数
        """
        if name in self._engines:
            logger.warning(f"数据库连接 {name} 已存在，将被替换")
        
        # 设置默认连接参数
        engine_kwargs = {
            "poolclass": QueuePool,
            "pool_size": kwargs.get("pool_size", 5),
            "max_overflow": kwargs.get("max_overflow", 10),
            "pool_timeout": kwargs.get("pool_timeout", 30),
            "pool_recycle": kwargs.get("pool_recycle", 3600),
            "echo": kwargs.get("echo", False),
        }
        
        # 创建引擎
        engine = create_engine(url, **engine_kwargs)
        
        # 添加连接事件监听器
        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            logger.debug(f"建立数据库连接: {name}")
        
        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            logger.debug(f"从连接池检出连接: {name}")
        
        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            logger.debug(f"连接返回连接池: {name}")
        
        # 创建会话工厂
        session_maker = sessionmaker(bind=engine)
        
        self._engines[name] = engine
        self._session_makers[name] = session_maker
        
        logger.info(f"添加数据库连接: {name} -> {url}")
    
    def get_engine(self, name: str = "default") -> Engine:
        """获取数据库引擎
        
        Args:
            name: 连接名称
            
        Returns:
            数据库引擎
            
        Raises:
            ValueError: 连接不存在
        """
        if not self._initialized:
            self.initialize()
        
        if name not in self._engines:
            raise ValueError(f"数据库连接 {name} 不存在")
        
        return self._engines[name]
    
    def get_session_maker(self, name: str = "default") -> sessionmaker:
        """获取会话工厂
        
        Args:
            name: 连接名称
            
        Returns:
            会话工厂
            
        Raises:
            ValueError: 连接不存在
        """
        if not self._initialized:
            self.initialize()
        
        if name not in self._session_makers:
            raise ValueError(f"数据库连接 {name} 不存在")
        
        return self._session_makers[name]
    
    @contextmanager
    def session(self, name: str = "default") -> Generator[Session, None, None]:
        """获取数据库会话上下文管理器
        
        Args:
            name: 连接名称
            
        Yields:
            数据库会话
        """
        if not self._initialized:
            self.initialize()
        
        session_maker = self.get_session_maker(name)
        session = session_maker()
        
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库会话异常: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self, name: str = "default") -> None:
        """创建所有表
        
        Args:
            name: 连接名称
        """
        engine = self.get_engine(name)
        Base.metadata.create_all(engine)
        logger.info(f"数据库 {name} 表创建完成")
    
    def drop_tables(self, name: str = "default") -> None:
        """删除所有表
        
        Args:
            name: 连接名称
        """
        engine = self.get_engine(name)
        Base.metadata.drop_all(engine)
        logger.info(f"数据库 {name} 表删除完成")
    
    def close_all(self) -> None:
        """关闭所有连接"""
        for name, engine in self._engines.items():
            engine.dispose()
            logger.info(f"关闭数据库连接: {name}")
        
        self._engines.clear()
        self._session_makers.clear()
        self._initialized = False


# 全局数据库连接管理器实例
_db_manager = None
_manager_lock = threading.Lock()


def get_db_manager() -> DatabaseConnectionManager:
    """获取全局数据库连接管理器实例
    
    Returns:
        数据库连接管理器实例
    """
    global _db_manager
    
    if _db_manager is None:
        with _manager_lock:
            if _db_manager is None:
                _db_manager = DatabaseConnectionManager()
    
    return _db_manager


def get_session(name: str = "default") -> Generator[Session, None, None]:
    """获取数据库会话上下文管理器
    
    Args:
        name: 连接名称
        
    Yields:
        数据库会话
    """
    db_manager = get_db_manager()
    with db_manager.session(name) as session:
        yield session


def init_database() -> None:
    """初始化数据库"""
    db_manager = get_db_manager()
    db_manager.initialize()
    db_manager.create_tables()


def close_database() -> None:
    """关闭数据库连接"""
    db_manager = get_db_manager()
    db_manager.close_all()