# myapp/logging_setup.py
import logging
import sys
import os
import socket
from ecs_logging import StdlibFormatter
import json
from datetime import datetime, timezone

_configured = False


APPLICATION_NAME = "kolegai"

def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


def get_machine_name() -> str:
    return _env("HOSTNAME", "POD_NAME", "COMPUTERNAME", default=socket.gethostname())


class ElasticFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self._ecs = StdlibFormatter(
            exclude_fields=[
                "ecs",
                "process",
                "log.origin",
                "log.original",
                "log.logger",
            ]
        )

        dummy = logging.LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None)
        self._standard = set(dummy.__dict__.keys())

        # Anything in here must NEVER go under "fields"
        self._no_fields = {
            "@timestamp", "timestamp",
            "level", "log.level",
            "message", "messageTemplate",
            "ecs", "ecs.version",
            "log", "process",
        }

    def format(self, record: logging.LogRecord) -> str:
        ecs_doc = json.loads(self._ecs.format(record))

        ts = ecs_doc.get("@timestamp") or datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        level = record.levelname.capitalize()

        message_template = record.msg if isinstance(record.msg, str) else str(record.msg)
        message = record.getMessage()

        fields: dict = {}

        # Copy extras, but never schema keys like "message"
        for k, v in record.__dict__.items():
            if k in self._standard:
                continue
            if k in self._no_fields:
                continue
            if k == "fields":  # reserved
                continue
            fields[k] = v

        # Allow extra={"fields": {...}} merge-in
        passed_fields = getattr(record, "fields", None)
        if isinstance(passed_fields, dict):
            # Also filter those keys, just in case
            for k, v in passed_fields.items():
                if k not in self._no_fields:
                    fields[k] = v

        fields.setdefault("SourceContext", record.name)
        fields.setdefault("Application", APPLICATION_NAME)
        fields.setdefault("MachineName", get_machine_name())

        out = {
            "@timestamp": ts,
            "level": level,
            "messageTemplate": message_template,
            "message": message,
            "fields": fields,
        }
        return json.dumps(out, ensure_ascii=False)


def configure_logging(source : str) -> None:
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(ElasticFormatter())  # ECS-compatible JSON

    root = logging.getLogger(source)
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging(name)
    return logging.getLogger(name)
