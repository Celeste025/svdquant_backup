#!/usr/bin/env python3
"""CPU regression for the H3 runtime hook contract, without loading H3 weights.

Extract the actual helper definitions by AST so DiffSynth/H3 pipeline imports do
not require a separate H3 environment. LowRankBranch is the actual vendored
DeepCompressor class. This checks PyTorch hook semantics, not model quality or
the fidelity of the NVFP4 codebook itself. No GPU is initialized by this script.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/deepcompressor"))
from deepcompressor.nn.patch.lowrank import LowRankBranch

SOURCE = ROOT / "scripts/minimax_h3_svdquant_common.py"
source_text = SOURCE.read_text()
names = {"_fp8_round_positive", "nvfp4_qdq", "DynamicActivationQDQ",
         "RuntimeHooks", "install_runtime_hooks", "temporary_quantized_linear"}
nodes = [node for node in ast.parse(source_text).body
         if getattr(node, "name", None) in names or
         (isinstance(node, ast.Assign) and any(
             isinstance(target, ast.Name) and target.id == "FP4_VALUES"
             for target in node.targets))]
scope = dict(torch=torch, nn=nn, Any=Any, dataclass=dataclass,
             contextmanager=contextmanager, LowRankBranch=LowRankBranch)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
qdq = scope["nvfp4_qdq"]
install = scope["install_runtime_hooks"]


def old_semantics_fixture(linear, smooth, a, b):
    """Frozen reproduction of the original pre-QDQ/post-branch hook ordering."""
    def pre(_module, args):
        x = args[0] / smooth if smooth is not None else args[0]
        return (qdq(x), *args[1:])

    def post(_module, inputs, output):
        return output + F.linear(F.linear(inputs[0], a), b)

    return [linear.register_forward_pre_hook(pre), linear.register_forward_hook(post)]


class RuntimeHookTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261002)
        self.linear = nn.Linear(16, 8, bias=True)
        self.a = torch.randn(3, 16)
        self.b = torch.randn(8, 3)
        self.smooth = torch.linspace(0.5, 1.5, 16)
        self.x = torch.randn(5, 16)

    def oracle(self, x, smooth, branch):
        z = x if smooth is None else x / smooth.to(x)
        y = F.linear(qdq(z), self.linear.weight, self.linear.bias)
        if branch is not None:
            y = y + F.linear(F.linear(z, branch.a.weight), branch.b.weight)
        return y

    def test_old_semantics_fixture_exposes_quantized_branch_input(self):
        handles = old_semantics_fixture(self.linear, self.smooth, self.a, self.b)
        try:
            z = self.x / self.smooth
            base = F.linear(qdq(z), self.linear.weight, self.linear.bias)
            actual = self.linear(self.x)
            wrong = base + F.linear(F.linear(qdq(z), self.a), self.b)
            correct = base + F.linear(F.linear(z, self.a), self.b)
            torch.testing.assert_close(actual, wrong, rtol=0, atol=0)
            self.assertGreater((actual - correct).abs().max().item(), 0.1)
        finally:
            for handle in handles:
                handle.remove()

    def test_high_precision_branch_bias_smoothing_and_repeated_calls(self):
        for dtype in (torch.float32, torch.bfloat16):
            self.linear.to(dtype)
            runtime = install(self.linear, self.smooth, self.a, self.b)
            observed = []
            capture = runtime.branch.register_forward_pre_hook(
                lambda _module, args: observed.append(args[0].detach().clone()))
            try:
                for x in (self.x.to(dtype), (self.x * 3.2).to(dtype),
                          self.x.reshape(1, 5, 16).to(dtype)):
                    actual = self.linear(x)
                    torch.testing.assert_close(actual, self.oracle(x, self.smooth, runtime.branch),
                                               rtol=0, atol=0)
                    torch.testing.assert_close(observed[-1], x / self.smooth.to(x), rtol=0, atol=0)
                    self.assertIsNone(runtime.act.branch_output)
                self.assertEqual(runtime.act.calls, 3)
            finally:
                capture.remove()
                runtime.remove()

    def test_no_smoothing(self):
        runtime = install(self.linear, None, self.a, self.b)
        try:
            torch.testing.assert_close(self.linear(self.x), self.oracle(self.x, None, runtime.branch),
                                       rtol=0, atol=0)
        finally:
            runtime.remove()

    def test_no_branch_keeps_plain_qdq_behavior(self):
        for smooth in (None, self.smooth):
            runtime = install(self.linear, smooth, None, None)
            try:
                torch.testing.assert_close(self.linear(self.x), self.oracle(self.x, smooth, None),
                                           rtol=0, atol=0)
                self.assertIsNone(runtime.branch)
                self.assertIsNone(runtime.act.branch_output)
            finally:
                runtime.remove()

    def test_branch_conversion_after_install_is_used(self):
        runtime = install(self.linear, self.smooth, self.a, self.b)
        # CPU float64 conversion checks that no weights/device/dtype were bound
        # into the pre-hook at installation. The actual H3 caller later uses CUDA.
        self.linear.to(torch.float64)
        runtime.branch.to(torch.float64)
        try:
            x = self.x.double()
            torch.testing.assert_close(self.linear(x), self.oracle(x, self.smooth, runtime.branch),
                                       rtol=0, atol=0)
        finally:
            runtime.remove()

    def test_main_forward_exception_clears_state_and_recovers(self):
        runtime = install(self.linear, self.smooth, self.a, self.b)
        original = self.linear.forward
        def fail(_x):
            raise ValueError("intentional main-forward failure")
        try:
            self.linear.forward = fail
            with self.assertRaisesRegex(ValueError, "main-forward"):
                self.linear(self.x)
            self.assertIsNone(runtime.act.branch_output)
            self.linear.forward = original
            torch.testing.assert_close(self.linear(self.x), self.oracle(self.x, self.smooth, runtime.branch),
                                       rtol=0, atol=0)
        finally:
            self.linear.forward = original
            runtime.remove()

    def test_pre_hook_exception_clears_state_and_recovers(self):
        runtime = install(self.linear, self.smooth, self.a, self.b)
        def fail(_module, _args):
            raise ValueError("intentional later pre-hook failure")
        failure = self.linear.register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(ValueError, "later pre-hook"):
                self.linear(self.x)
            self.assertIsNone(runtime.act.branch_output)
            failure.remove()
            torch.testing.assert_close(self.linear(self.x), self.oracle(self.x, self.smooth, runtime.branch),
                                       rtol=0, atol=0)
        finally:
            failure.remove()
            runtime.remove()

    def test_branch_exception_clears_state_and_recovers(self):
        runtime = install(self.linear, self.smooth, self.a, self.b)
        original = runtime.branch.forward
        def fail(_x):
            raise ValueError("intentional branch failure")
        try:
            runtime.branch.forward = fail
            with self.assertRaisesRegex(ValueError, "branch failure"):
                self.linear(self.x)
            self.assertIsNone(runtime.act.branch_output)
            runtime.branch.forward = original
            torch.testing.assert_close(self.linear(self.x), self.oracle(self.x, self.smooth, runtime.branch),
                                       rtol=0, atol=0)
        finally:
            runtime.branch.forward = original
            runtime.remove()

    def test_remove_clears_pending_output_and_is_idempotent(self):
        runtime = install(self.linear, self.smooth, self.a, self.b)
        runtime.act(self.linear, (self.x,))
        self.assertIsNotNone(runtime.act.branch_output)
        runtime.remove()
        runtime.remove()
        self.assertIsNone(runtime.act.branch_output)
        self.assertEqual(runtime.handles, [])
        self.assertEqual(len(self.linear._forward_pre_hooks), 0)
        self.assertEqual(len(self.linear._forward_hooks), 0)
        torch.testing.assert_close(self.linear(self.x), F.linear(self.x, self.linear.weight, self.linear.bias),
                                   rtol=0, atol=0)

    def test_temporary_quantized_linear_restores_on_exception(self):
        weight = self.linear.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "body failure"):
            with scope["temporary_quantized_linear"](
                self.linear, weight * 0.8, self.smooth, self.a, self.b
            ) as runtime:
                torch.testing.assert_close(self.linear(self.x), self.oracle(self.x, self.smooth, runtime.branch),
                                           rtol=0, atol=0)
                raise ValueError("body failure")
        torch.testing.assert_close(self.linear.weight, weight, rtol=0, atol=0)
        self.assertIsNone(runtime.act.branch_output)
        self.assertEqual(len(self.linear._forward_hooks), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeHookTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": [name.id() for name, _ in result.failures],
        "errors": [name.id() for name, _ in result.errors],
        "device": "cpu", "torch": torch.__version__,
        "source": str(SOURCE), "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "boundary": "AST-loaded actual helpers; actual vendored LowRankBranch; no H3 weights or GPU; not a quality test",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
