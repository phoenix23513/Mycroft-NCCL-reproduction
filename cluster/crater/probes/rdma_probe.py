#!/usr/bin/env python3
"""M1 本地资源预检；不执行 collective，也不证明跨节点 RDMA 已连通。"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess


def command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=20)
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"exit_code": None, "error": str(error)}


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def verbs_devices():
    """实际加载 verbs 并打开设备 context；不注册 MR、不建立 QP 或传输数据。"""
    try:
        library = ctypes.CDLL("libibverbs.so.1", use_errno=True)
        library.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
        library.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.ibv_free_device_list.restype = None
        library.ibv_get_device_name.argtypes = [ctypes.c_void_p]
        library.ibv_get_device_name.restype = ctypes.c_char_p
        library.ibv_open_device.argtypes = [ctypes.c_void_p]
        library.ibv_open_device.restype = ctypes.c_void_p
        library.ibv_close_device.argtypes = [ctypes.c_void_p]
        library.ibv_close_device.restype = ctypes.c_int
    except (OSError, AttributeError) as error:
        return {"loaded": False, "error": str(error), "devices": []}
    count = ctypes.c_int()
    devices = library.ibv_get_device_list(ctypes.byref(count))
    if not devices:
        return {"loaded": True, "errno": ctypes.get_errno(), "devices": []}
    records = []
    try:
        for index in range(count.value):
            name = library.ibv_get_device_name(devices[index]).decode()
            ctypes.set_errno(0)
            context = library.ibv_open_device(devices[index])
            record = {"name": name, "context_opened": bool(context)}
            if context:
                record["close_exit_code"] = library.ibv_close_device(context)
            else:
                record["errno"] = ctypes.get_errno()
            records.append(record)
    finally:
        library.ibv_free_device_list(devices)
    return {"loaded": True, "devices": records}


def probe(rank):
    ports = []
    for port in sorted(Path("/sys/class/infiniband").glob("*/ports/*")):
        ports.append({"device": port.parent.parent.name, "port": port.name,
                      "state": read(port / "state"), "physical_state": read(port / "phys_state"),
                      "link_layer": read(port / "link_layer"), "rate": read(port / "rate")})
    identifiers = {}
    for label, path in (("dmi_product_uuid", "/sys/class/dmi/id/product_uuid"),
                        ("kernel_boot_id", "/proc/sys/kernel/random/boot_id")):
        value = read(path)
        identifiers[label + "_sha256"] = hashlib.sha256(value.encode()).hexdigest() if value else None
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    gpu = command(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version", "--format=csv"])
    verbs = verbs_devices()
    opened = {item["name"] for item in verbs["devices"] if item["context_opened"]}
    active = [item for item in ports if item["device"] in opened and
              (item["state"] or "").startswith("4:") and
              item["link_layer"] in ("InfiniBand", "Ethernet")]
    blocked = []
    if gpu["exit_code"] != 0:
        blocked.append("nvidia-smi failed or missing")
    if not verbs["loaded"]:
        blocked.append("libibverbs.so.1 cannot be loaded")
    if not opened:
        blocked.append("no usable verbs device context")
    if not active:
        blocked.append("no active IB/RoCE port on an opened verbs device")
    if soft == 0:
        blocked.append("memlock soft limit is zero")
    return {"probe": "m1_rdma_resources", "rank": rank, "hostname": socket.gethostname(),
            "node_identifiers": identifiers, "kernel": platform.release(),
            "os_release": read("/etc/os-release"),
            "memlock_bytes": {"soft": soft, "hard": hard, "unlimited_value": resource.RLIM_INFINITY},
            "device_files": [str(p) for p in sorted(Path("/dev/infiniband").glob("*"))],
            "verbs": verbs, "ports": ports, "gpu_inventory": gpu,
            "compiler": command(["g++", "--version"]),
            "cuda_compiler": command(["nvcc", "--version"]),
            "network_interfaces": command(["ip", "-brief", "address"]),
            "ibv_devinfo": command(["ibv_devinfo"]),
            "environment": {key: os.environ[key] for key in
                            ("CUDA_VISIBLE_DEVICES", "NCCL_IB_HCA", "NCCL_SOCKET_IFNAME") if key in os.environ},
            "blocked_reasons": blocked,
            "status": "BLOCKED" if blocked else "RESOURCES_VISIBLE",
            "scope": "local_resource_precheck_only; not_cross_node_RDMA_acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    result = probe(args.rank)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(2 if result["blocked_reasons"] else 0)


if __name__ == "__main__":
    main()
