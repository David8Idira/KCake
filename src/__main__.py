"""
KCake 主入口

异构设备集群化超大规模模型推理引擎
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from .core.inference_engine import InferenceEngine
from .api.server import KcakeAPIServer
from .cluster.manager import ClusterManager, ClusterConfig, NodeRole
from .heterogeneous.scheduler import HeteroScheduler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        description="KCake - 异构设备集群化超大规模模型推理引擎"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # serve命令 - 启动API服务
    serve_parser = subparsers.add_parser("serve", help="启动API服务")
    serve_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称或路径"
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="监听地址"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="监听端口"
    )
    serve_parser.add_argument(
        "--cluster-key",
        type=str,
        default=None,
        help="集群密钥"
    )
    serve_parser.add_argument(
        "--name",
        type=str,
        default="kcake-node",
        help="节点名称"
    )
    serve_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "metal", "vulkan"],
        help="设备类型"
    )
    serve_parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="数据类型"
    )
    
    # worker命令 - 启动工作节点
    worker_parser = subparsers.add_parser("worker", help="启动工作节点")
    worker_parser.add_argument(
        "--cluster-key",
        type=str,
        required=True,
        help="集群密钥"
    )
    worker_parser.add_argument(
        "--master-host",
        type=str,
        required=True,
        help="Master节点地址"
    )
    worker_parser.add_argument(
        "--master-port",
        type=int,
        default=8000,
        help="Master节点端口"
    )
    worker_parser.add_argument(
        "--name",
        type=str,
        default="kcake-worker",
        help="节点名称"
    )
    
    # run命令 - 运行单次推理
    run_parser = subparsers.add_parser("run", help="运行单次推理")
    run_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称或路径"
    )
    run_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="输入提示"
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="最大生成token数"
    )
    
    # chat命令 - 交互式聊天
    chat_parser = subparsers.add_parser("chat", help="交互式聊天")
    chat_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称或路径"
    )
    
    return parser


async def serve_command(args: argparse.Namespace) -> int:
    """serve命令实现"""
    logger.info(f"启动KCake API服务: {args.host}:{args.port}")
    
    # 初始化推理引擎
    engine = InferenceEngine(device=args.device)
    
    # 加载模型
    if args.model:
        logger.info(f"加载模型: {args.model}")
        success = await engine.load_model(
            args.model,
            torch_dtype=args.dtype
        )
        
        if not success:
            logger.error("模型加载失败")
            return 1
        
        logger.info("模型加载成功")
    
    # 初始化集群管理器
    cluster_manager = None
    if args.cluster_key:
        config = ClusterConfig(
            cluster_key=args.cluster_key,
            master_host=args.host,
            master_port=args.port,
            node_name=args.name,
            node_role=NodeRole.MASTER
        )
        cluster_manager = ClusterManager(config)
        await cluster_manager.start()
    
    # 初始化API服务器
    server = KcakeAPIServer(
        inference_engine=engine,
        cluster_manager=cluster_manager,
        host=args.host,
        port=args.port
    )
    
    # 启动服务器
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    finally:
        if cluster_manager:
            await cluster_manager.stop()
        await engine.unload_model()
    
    return 0


async def worker_command(args: argparse.Namespace) -> int:
    """worker命令实现"""
    logger.info(f"启动KCake Worker节点: {args.name}")
    
    # 初始化推理引擎
    engine = InferenceEngine(device="cpu")
    
    # 初始化集群管理器
    config = ClusterConfig(
        cluster_key=args.cluster_key,
        master_host=args.master_host,
        master_port=args.master_port,
        node_name=args.name,
        node_role=NodeRole.WORKER
    )
    cluster_manager = ClusterManager(config)
    
    # 启动
    try:
        await cluster_manager.start()
        
        # 保持运行
        while True:
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    finally:
        await cluster_manager.stop()
        await engine.unload_model()
    
    return 0


async def run_command(args: argparse.Namespace) -> int:
    """run命令实现"""
    logger.info(f"运行推理: {args.model}")
    
    # 初始化引擎
    engine = InferenceEngine(device="cpu")
    
    # 加载模型
    success = await engine.load_model(args.model)
    if not success:
        logger.error("模型加载失败")
        return 1
    
    # 创建请求
    from .core.inference_engine import InferenceRequest
    request = InferenceRequest(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens
    )
    
    # 执行推理
    response = await engine.generate(request)
    
    # 输出结果
    print("\n" + "=" * 50)
    print("推理结果:")
    print("=" * 50)
    print(response.text)
    print("=" * 50)
    print(f"生成token数: {response.tokens_generated}")
    print(f"完成原因: {response.finish_reason}")
    
    # 清理
    await engine.unload_model()
    
    return 0


async def chat_command(args: argparse.Namespace) -> int:
    """chat命令实现"""
    logger.info(f"启动交互式聊天: {args.model}")
    
    # 初始化引擎
    engine = InferenceEngine(device="cpu")
    
    # 加载模型
    success = await engine.load_model(args.model)
    if not success:
        logger.error("模型加载失败")
        return 1
    
    print("\n" + "=" * 50)
    print(f"KCake 聊天模式 (模型: {args.model})")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50 + "\n")
    
    # 交互式循环
    while True:
        try:
            prompt = input("User: ").strip()
            
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            
            if not prompt:
                continue
            
            # 创建请求
            from .core.inference_engine import InferenceRequest
            request = InferenceRequest(
                model=args.model,
                prompt=f"User: {prompt}\n\nAssistant: ",
                max_tokens=512
            )
            
            # 执行推理
            response = await engine.generate(request)
            
            print(f"\nAssistant: {response.text}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n错误: {e}\n")
    
    # 清理
    await engine.unload_model()
    print("\n再见!")
    
    return 0


def main() -> int:
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # 执行相应命令
    if args.command == "serve":
        return asyncio.run(serve_command(args))
    elif args.command == "worker":
        return asyncio.run(worker_command(args))
    elif args.command == "run":
        return asyncio.run(run_command(args))
    elif args.command == "chat":
        return asyncio.run(chat_command(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
