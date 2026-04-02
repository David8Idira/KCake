#!/usr/bin/env python3
"""
KCake CLI - 命令行入口

用法:
    python -m kcake serve --model <model_name> --cluster-key <key>
    python -m kcake worker --cluster-key <key> --name <node_name>
    python -m kcake run --model <model_name> <prompt>
    python -m kcake chat --model <model_name>
    python -m kcake status
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from . import __version__
from .core import InferenceEngine, InferenceRequest, DeviceType
from .heterogeneous import HeteroScheduler, PlacementPolicy
from .cluster import ClusterManager, ClusterConfig, NodeRole
from .api import APIServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_serve(args):
    """启动 API 服务器 (主节点)"""
    async def main():
        # 初始化推理引擎
        engine = InferenceEngine(
            model_path=args.model,
            device_map=args.device_map or "auto",
            max_memory=args.max_memory,
            torch_dtype=args.dtype or "float16",
            quantization=args.quantization,
        )
        
        # 初始化异构调度器
        scheduler = HeteroScheduler(
            placement_policy=PlacementPolicy(
                hot_threshold=0.8,
                warm_threshold=0.3,
                gpu_memory_ratio=0.9,
            ),
            enable_dynamic_migration=True,
            numa_aware=True,
        )
        
        # 设置调度器
        engine.set_heterogeneous_scheduler(scheduler)
        
        # 初始化集群管理器
        cluster_config = ClusterConfig(
            cluster_key=args.cluster_key,
            master_name=args.name or "master",
            mdns_enabled=not args.no_mdns,
        )
        
        cluster = ClusterManager(
            config=cluster_config,
            local_node_name=args.name or "master",
            local_role=NodeRole.MASTER,
            local_host=args.host or "0.0.0.0",
            local_port=args.port or 8000,
        )
        
        # 初始化 API 服务器
        api = APIServer(
            host=args.host or "0.0.0.0",
            port=args.port or 8000,
            engine=engine,
            cluster_manager=cluster,
        )
        
        # 启动
        await cluster.start()
        await api.start()
    
    asyncio.run(main())


def cmd_worker(args):
    """启动 Worker 节点"""
    async def main():
        cluster_config = ClusterConfig(
            cluster_key=args.cluster_key,
            mdns_enabled=not args.no_mdns,
        )
        
        cluster = ClusterManager(
            config=cluster_config,
            local_node_name=args.name,
            local_role=NodeRole.WORKER,
            local_host=args.host or "localhost",
            local_port=args.port or 8000,
        )
        
        # 加入主节点集群
        if args.master_host and args.master_port:
            success = await cluster.join_cluster(args.master_host, args.master_port)
            if not success:
                logger.error("Failed to join cluster")
                return
        
        await cluster.start()
        
        # 保持运行
        while True:
            await asyncio.sleep(1)
    
    asyncio.run(main())


def cmd_run(args):
    """运行单次推理"""
    async def main():
        engine = InferenceEngine(
            model_path=args.model,
            quantization=args.quantization,
        )
        
        request = InferenceRequest(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens or 2048,
            temperature=args.temperature or 0.7,
        )
        
        async for token in engine.generate(request):
            print(token, end="", flush=True)
        
        print()
    
    asyncio.run(main())


def cmd_chat(args):
    """交互式聊天"""
    async def main():
        engine = InferenceEngine(
            model_path=args.model,
            quantization=args.quantization,
        )
        
        print(f"KCake Chat - Model: {args.model}")
        print("Type 'exit' or 'quit' to end the conversation")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            if not user_input:
                continue
            
            request = InferenceRequest(
                model=args.model,
                prompt=f"User: {user_input}\nAssistant:",
                max_tokens=args.max_tokens or 2048,
                temperature=args.temperature or 0.7,
            )
            
            print("\nAssistant: ", end="", flush=True)
            async for token in engine.generate(request):
                print(token, end="", flush=True)
            print()
    
    asyncio.run(main())


def cmd_status(args):
    """显示状态"""
    print(f"KCake v{__version__}")
    print()
    print("To check cluster status, run: python -m kcake serve --cluster-key <key>")
    print("Then access http://localhost:8000/cluster/status")


def cmd_worker_list(args):
    """列出附近的工作节点 (通过 mDNS)"""
    async def main():
        cluster_config = ClusterConfig(
            cluster_key=args.cluster_key,
            mdns_enabled=True,
        )
        
        cluster = ClusterManager(
            config=cluster_config,
            local_node_name="scanner",
            local_role=NodeRole.HYBRID,
        )
        
        print(f"Scanning for KCake nodes with cluster key: {args.cluster_key[:4]}***")
        
        # 触发扫描
        await cluster._scan_for_nodes()
        
        # 等待结果
        await asyncio.sleep(5)
        
        status = cluster.get_cluster_status()
        print(f"\nFound {status['total_nodes']} nodes:")
        for node_id, node_info in status.get("nodes", {}).items():
            print(f"  - {node_info['name']} ({node_info['role']}) - {node_info['status']}")
    
    asyncio.run(main())


def main():
    parser = argparse.ArgumentParser(
        description="KCake - 异构设备集群化超大规模模型推理引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--version", action="version", version=f"KCake {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # serve 命令
    serve_parser = subparsers.add_parser("serve", help="启动 API 服务器 (主节点)")
    serve_parser.add_argument("--model", required=True, help="模型名称或路径")
    serve_parser.add_argument("--cluster-key", required=True, help="集群认证密钥")
    serve_parser.add_argument("--name", default="master", help="节点名称")
    serve_parser.add_argument("--host", default="0.0.0.0", help="监听主机")
    serve_parser.add_argument("--port", type=int, default=8000, help="监听端口")
    serve_parser.add_argument("--device-map", help="设备映射 (auto, cuda, cpu)")
    serve_parser.add_argument("--max-memory", type=lambda x: dict(m.split(":") for m in x.split(",")), help="最大内存配置")
    serve_parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="数据类型")
    serve_parser.add_argument("--quantization", choices=["int4", "int8", "fp8", "none"], help="量化类型")
    serve_parser.add_argument("--no-mdns", action="store_true", help="禁用 mDNS 发现")
    serve_parser.set_defaults(func=cmd_serve)
    
    # worker 命令
    worker_parser = subparsers.add_parser("worker", help="启动 Worker 节点")
    worker_parser.add_argument("--cluster-key", required=True, help="集群认证密钥")
    worker_parser.add_argument("--name", required=True, help="节点名称")
    worker_parser.add_argument("--master-host", help="主节点主机")
    worker_parser.add_argument("--master-port", type=int, help="主节点端口")
    worker_parser.add_argument("--host", default="localhost", help="监听主机")
    worker_parser.add_argument("--port", type=int, default=8001, help="监听端口")
    worker_parser.add_argument("--no-mdns", action="store_true", help="禁用 mDNS 发现")
    worker_parser.set_defaults(func=cmd_worker)
    
    # run 命令
    run_parser = subparsers.add_parser("run", help="运行单次推理")
    run_parser.add_argument("prompt", help="输入提示词")
    run_parser.add_argument("--model", required=True, help="模型名称或路径")
    run_parser.add_argument("--max-tokens", type=int, default=2048, help="最大生成 token 数")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    run_parser.add_argument("--quantization", choices=["int4", "int8", "fp8", "none"], help="量化类型")
    run_parser.set_defaults(func=cmd_run)
    
    # chat 命令
    chat_parser = subparsers.add_parser("chat", help="交互式聊天")
    chat_parser.add_argument("--model", required=True, help="模型名称或路径")
    chat_parser.add_argument("--max-tokens", type=int, default=2048, help="最大生成 token 数")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    chat_parser.add_argument("--quantization", choices=["int4", "int8", "fp8", "none"], help="量化类型")
    chat_parser.set_defaults(func=cmd_chat)
    
    # status 命令
    status_parser = subparsers.add_parser("status", help="显示状态")
    status_parser.set_defaults(func=cmd_status)
    
    # worker-list 命令
    worker_list_parser = subparsers.add_parser("worker-list", help="列出附近的工作节点")
    worker_list_parser.add_argument("--cluster-key", required=True, help="集群认证密钥")
    worker_list_parser.set_defaults(func=cmd_worker_list)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
