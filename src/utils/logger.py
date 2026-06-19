"""CustomLogger — 项目统一日志管理器

基于 loguru 构建，提供模块级日志隔离与统一管理。

特性
────
- 每个模块自动拥有独立的日志文件（`logs/<模块名>.log`）
- 控制台彩色输出，便于开发调试
- 文件日志自动轮转（单文件 10 MB，保留最近 7 天）
- 线程安全，支持多模块并发写日志
- 全局配置一次，全项目所有模块统一使用

快速开始
────────
.. code-block:: python

    from utils.logger import CustomLogger

    # 可选：全局配置（未调用时使用默认值）
    CustomLogger.configure(level="DEBUG", log_dir="logs")

    # 获取模块专属 logger
    log = CustomLogger.get_logger(__name__)
    log.info("模型合并完成")
    log.error("导出失败: {}", reason)

参考
────
- loguru 文档: https://github.com/Delgan/loguru
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger


class CustomLogger:
    """项目统一日志管理器。

    所有模块通过 ``CustomLogger.get_logger(__name__)`` 获取专属 logger，
    日志会同时输出到控制台与 ``logs/<模块名>.log`` 文件。

    Attributes:
        _configured: 是否已完成全局配置
        _log_dir: 日志文件输出目录
        _registered_files: 已注册的文件 handler 集合（防止重复添加）
    """

    _configured: bool = False
    _log_dir: Optional[Path] = None
    _registered_files: set[str] = set()

    # ── 全局配置 ──────────────────────────────────────────

    @classmethod
    def configure(
        cls,
        log_dir: Optional[str] = None,
        level: str = "INFO",
        *,
        rotation: str = "10 MB",
        retention: str = "7 days",
    ) -> None:
        """一次性全局配置日志系统。

        只需调用一次（在应用入口或最先加载的模块中）。未调用时
        ``get_logger`` 会自动使用默认配置。

        Args:
            log_dir: 日志输出目录，默认 ``<项目根>/logs``
            level: 控制台最低输出级别（DEBUG/INFO/WARNING/ERROR）
            rotation: 文件轮转条件（如 ``"10 MB"``, ``"1 day"``）
            retention: 旧日志保留时长（如 ``"7 days"``, ``"1 month"``）
        """
        if cls._configured:
            return

        # 定位项目根：utils/logger.py → src/ → 项目根
        _project_root = Path(__file__).resolve().parent.parent.parent
        cls._log_dir = Path(log_dir) if log_dir else _project_root / "logs"
        cls._log_dir.mkdir(parents=True, exist_ok=True)

        # 移除 loguru 默认 handler
        _loguru_logger.remove()

        # 控制台 handler —— 彩色、紧凑
        _loguru_logger.add(
            sys.stderr,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[module]}</cyan> - "
                "<level>{message}</level>"
            ),
            level=level.upper(),
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        cls._configured = True
        cls._rotation = rotation
        cls._retention = retention

    # ── 模块 Logger ──────────────────────────────────────

    @classmethod
    def get_logger(cls, name: str = "copaw-dpo"):
        """获取模块专属 logger。

        首次调用时会自动完成全局配置。同一 ``name`` 重复调用
        返回同一个 contextualized logger 实例。

        Args:
            name: 模块标识名（建议传入 ``__name__``）

        Returns:
            绑定 ``module`` 上下文的 loguru logger

        Example:
            >>> log = CustomLogger.get_logger(__name__)
            >>> log.info("开始合并模型")
        """
        if not cls._configured:
            cls.configure()

        # 清理模块名：保留最后两段（如 "m_merge.exporter"）
        parts = name.split(".")
        short_name = ".".join(parts[-2:]) if len(parts) >= 2 else name
        safe_name = short_name.replace(".", "_")

        # 为此模块注册专属文件 handler（仅一次）
        log_file = cls._log_dir / f"{safe_name}.log"
        file_key = str(log_file.resolve())
        if file_key not in cls._registered_files:
            _loguru_logger.add(
                str(log_file),
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                    "{level: <8} | "
                    "{name}:{function}:{line} - "
                    "{message}"
                ),
                level="DEBUG",
                rotation=cls._rotation,
                retention=cls._retention,
                encoding="utf-8",
                enqueue=True,  # 线程安全
            )
            cls._registered_files.add(file_key)

        return _loguru_logger.bind(module=short_name)

    # ── 便捷方法 ─────────────────────────────────────────

    @classmethod
    def info(cls, message: str, *args, **kwargs) -> None:
        """全局级 info 日志（用于非模块化的入口脚本）。"""
        cls.get_logger("copaw-dpo").info(message, *args, **kwargs)

    @classmethod
    def warning(cls, message: str, *args, **kwargs) -> None:
        """全局级 warning 日志。"""
        cls.get_logger("copaw-dpo").warning(message, *args, **kwargs)

    @classmethod
    def error(cls, message: str, *args, **kwargs) -> None:
        """全局级 error 日志。"""
        cls.get_logger("copaw-dpo").error(message, *args, **kwargs)

    @classmethod
    def debug(cls, message: str, *args, **kwargs) -> None:
        """全局级 debug 日志。"""
        cls.get_logger("copaw-dpo").debug(message, *args, **kwargs)

    @classmethod
    def exception(cls, message: str, *args, **kwargs) -> None:
        """全局级 exception 日志（含调用栈）。"""
        cls.get_logger("copaw-dpo").exception(message, *args, **kwargs)
