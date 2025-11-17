# ai_record_processor/src/__init__.py

"""
AI Record Processor — A fault-tolerant, high-performance Ray Data Processor
that performs inference by dispatching RPC requests to dynamically registered
inference actors (e.g., vLLM servers, custom Triton servers, etc.).

Features
--------
- Dynamic service discovery & auto-healing registry
- Exponential backoff + retry + timeout
- Service list caching + live-ness probing
- Round-robin / random load balancing
- Fully compatible with Ray Data OfflineProcessor / ProcessorBuilder
- Hot add/remove of inference actors without stopping the pipeline
"""

from .registry import ServiceRegistry
from .config import AIRecordProcessorConfig
from .stage import RPCInferenceStage
from .processor import build_ai_record_processor

# Convenience exports — 用户只需要 from ai_record_processor import XXX 即可
__all__ = [
    "ServiceRegistry",
    "AIRecordProcessorConfig",
    "RPCInferenceStage",
    "build_ai_record_processor",
]

# Package metadata
__version__ = "0.1.0"
__author__ = "Your Team"
__description__ = "Fault-tolerant RPC-based inference processor for Ray Data"

# Optional: 自动注册到 ProcessorBuilder（只需 import 这个包就会注册）
try:
    # 只有在 Ray 环境中才会成功导入并注册，避免单元测试时出错
    from .processor import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass