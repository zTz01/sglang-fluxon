#!/usr/bin/env python3
"""Behavior checks for Fluxon hostless write-through and write-back ancestry."""
from __future__ import annotations

import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/mem_cache/unified_radix_cache.py"
)
BASE_COMPONENT_TYPE = 0
logger = logging.getLogger(__name__)


class Node:
    def __init__(self, node_id: int, parent: "Node | None", tokens: int = 64):
        self.id = node_id
        self.parent = parent
        self.ready = False
        self.component_data = {
            BASE_COMPONENT_TYPE: SimpleNamespace(
                value=list(range(tokens)),
                metadata={},
            )
        }


class CacheFixture:
    def __init__(self):
        self.root_node = Node(0, None, tokens=0)
        self.cache_controller = object()
        self.ongoing_fluxon_hostless_backup = {}
        self.submissions: list[tuple[int, bool]] = []
        self.waited_nodes: list[int] = []

    def _is_fluxon_hostless_full_mode(self) -> bool:
        return True

    def _fluxon_hostless_node_storage_ready(self, node: Node) -> bool:
        return node.ready

    def _has_pending_fluxon_hostless_ack(self, node_id: int) -> bool:
        return False

    def _fluxon_backend(self):
        return object()

    def _submit_fluxon_hostless_write_batch(
        self, node: Node, backend, *, write_back: bool
    ):
        self.submissions.append((node.id, write_back))
        node.ready = True
        return SimpleNamespace(dedicated_write_stream=False)

    def _wait_for_fluxon_hostless_backup(self, node: Node) -> None:
        self.waited_nodes.append(node.id)


def load_write_backup_method():
    tree = ast.parse(SOURCE.read_bytes(), filename=str(SOURCE))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnifiedRadixCache"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "write_backup"
    )
    shell = ast.parse("class TestedCache(CacheFixture):\n    pass\n").body[0]
    shell.body = [copy.deepcopy(method)]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            shell,
        ],
        type_ignores=[],
    )
    namespace = {
        "CacheFixture": CacheFixture,
        "BASE_COMPONENT_TYPE": BASE_COMPONENT_TYPE,
        "logger": logger,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["TestedCache"]


TestedCache = load_write_backup_method()


class FluxonWriteBackupPolicyTest(unittest.TestCase):
    def make_path(self):
        cache = TestedCache()
        parent = Node(1, cache.root_node)
        child = Node(2, parent)
        return cache, parent, child

    def test_write_through_closes_unbacked_ancestor_prefix(self):
        cache, parent, child = self.make_path()

        written = cache.write_backup(child, write_back=False)

        self.assertEqual(written, 64)
        self.assertEqual(cache.submissions, [(parent.id, False), (child.id, False)])
        self.assertTrue(parent.ready)
        self.assertTrue(child.ready)
        self.assertEqual(cache.waited_nodes, [])

    def test_write_back_submits_only_the_current_victim(self):
        cache, parent, child = self.make_path()

        written = cache.write_backup(child, write_back=True)

        self.assertEqual(written, 64)
        self.assertEqual(cache.submissions, [(child.id, True)])
        self.assertFalse(parent.ready)
        self.assertTrue(child.ready)
        self.assertEqual(cache.waited_nodes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
