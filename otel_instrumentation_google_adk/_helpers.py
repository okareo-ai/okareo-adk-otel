import base64
import inspect
import json
from collections import OrderedDict
from typing import Any


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize obj to a JSON-formatted string, falling back to str() for
    un-serializable values so that the call never raises."""
    return json.dumps(obj, default=_default, ensure_ascii=False, **kwargs)


def bind_args_kwargs(func: Any, *args: Any, **kwargs: Any) -> OrderedDict[str, Any]:
    """Bind *args* and **kwargs* to *func*'s signature and return the resolved arguments."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _default(obj: Any) -> Any:
    """JSON serializer fallback for pydantic models, classes, and bytes."""
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    if inspect.isclass(obj) and issubclass(obj, BaseModel):
        return obj.model_json_schema()
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode()
    return str(obj)
