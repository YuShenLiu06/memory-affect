"""Static dependency/ownership checks; UI and host loop must stay outside core."""

import ast
from pathlib import Path


def test_core_does_not_import_ui_vendor_clients_or_agent_harnesses():
    root = Path(__file__).resolve().parents[1] / "src" / "memory_affect"
    forbidden = {"streamlit", "openai", "anthropic", "pi", "dsh", "schedule", "apscheduler"}
    for source in root.glob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [(node.module or "").split(".")[0]]
            else:
                continue
            assert forbidden.isdisjoint(modules), (source.name, modules)


def test_core_does_not_create_background_threads_or_timer_jobs():
    root = Path(__file__).resolve().parents[1] / "src" / "memory_affect"
    for source in root.glob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            forbidden = {"Thread", "Timer", "BackgroundScheduler", "create_task"}
            assert name not in forbidden, source.name
