#!/usr/bin/env python3
"""Spawn a deployed Talkie export job without tying it to a local Modal app run."""

from __future__ import annotations

import argparse

import modal


APP_NAME = "onnx-webgpu-export-job"
GPU_FUNCTION_NAME = "run_export_mode"
CPU_FUNCTION_NAME = "run_export_mode_cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["q4", "q8", "q8-fp16", "q8-fold", "q8-from-folded", "split-upload", "upload", "all"])
    parser.add_argument("--h100", action="store_true", help="Force the H100 function even for q8/upload.")
    parser.add_argument("--wait", action="store_true", help="Keep this process attached until the remote call exits.")
    args = parser.parse_args()

    gpu_modes = {"q4", "q8-fp16", "q8-fold", "q8-from-folded", "all"}
    function_name = GPU_FUNCTION_NAME if args.h100 or args.mode in gpu_modes else CPU_FUNCTION_NAME
    function = modal.Function.from_name(APP_NAME, function_name)
    call = function.spawn(args.mode)
    print(f"Spawned Talkie export mode={args.mode} function={function_name} call_id={call.object_id}", flush=True)
    print(call.get_dashboard_url(), flush=True)
    if args.wait:
        call.get()
        print("Remote call completed.", flush=True)


if __name__ == "__main__":
    main()
