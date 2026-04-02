"""
KCake Tests - 测试套件
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


class TestHeteroScheduler:
    """异构调度器测试"""
    
    def test_expert_classification(self):
        """测试专家分类"""
        from kcake.heterogeneous import HeteroScheduler, PlacementPolicy, ExpertType
        
        scheduler = HeteroScheduler(
            placement_policy=PlacementPolicy(
                hot_threshold=0.8,
                warm_threshold=0.3,
            )
        )
        
        # 测试高频专家
        hot_expert = scheduler._classify_expert(0.9)
        assert hot_expert == ExpertType.HOT
        
        # 测试中频专家
        warm_expert = scheduler._classify_expert(0.5)
        assert warm_expert == ExpertType.WARM
        
        # 测试低频专家
        cold_expert = scheduler._classify_expert(0.1)
        assert cold_expert == ExpertType.COLD
    
    def test_device_registration(self):
        """测试设备注册"""
        from kcake.heterogeneous import HeteroScheduler
        
        scheduler = HeteroScheduler()
        
        scheduler.register_device(
            device_id="cuda:0",
            device_type="cuda",
            total_memory=24 * (1024**3),  # 24GB
            numa_node=0,
        )
        
        assert "cuda:0" in scheduler.device_memories
        assert scheduler.device_memories["cuda:0"].device_type == "cuda"
    
    @pytest.mark.asyncio
    async def test_expert_scheduling(self):
        """测试专家调度"""
        from kcake.heterogeneous import HeteroScheduler, Expert
        
        scheduler = HeteroScheduler()
        
        # 注册设备
        scheduler.register_device(
            device_id="cuda:0",
            device_type="cuda",
            total_memory=24 * (1024**3),
        )
        
        scheduler.register_device(
            device_id="cpu",
            device_type="cpu",
            total_memory=128 * (1024**3),
        )
        
        # 注册专家
        scheduler.experts[0] = Expert(
            expert_id=0,
            name="expert_0",
            parameters=1_000_000_000,
            call_frequency=0.9,
            memory_usage_bytes=2 * (1024**3),  # 2GB
        )
        
        # 调度
        schedule = await scheduler.schedule_experts([0], ["cuda:0", "cpu"])
        
        assert 0 in schedule
        assert schedule[0] in ["cuda:0", "cpu"]


class TestClusterManager:
    """集群管理器测试"""
    
    def test_cluster_config(self):
        """测试集群配置"""
        from kcake.cluster import ClusterConfig, NodeRole, NodeStatus
        
        config = ClusterConfig(
            cluster_key="test_key_123",
            master_name="test_master",
            mdns_enabled=True,
        )
        
        assert config.cluster_key == "test_key_123"
        assert config.master_name == "test_master"
        assert config.mdns_enabled is True
    
    def test_node_info(self):
        """测试节点信息"""
        from kcake.cluster import NodeInfo, NodeRole, NodeStatus
        
        node = NodeInfo(
            node_id="test_node",
            name="test_node_1",
            role=NodeRole.MASTER,
            host="localhost",
            port=8000,
            device_type="cuda",
            memory_total=24 * (1024**3),
            memory_available=20 * (1024**3),
            compute_score=1.0,
        )
        
        assert node.is_available is True
        assert node.endpoint == "http://localhost:8000"


class TestInferenceEngine:
    """推理引擎测试"""
    
    def test_device_type_enum(self):
        """测试设备类型枚举"""
        from kcake.core import DeviceType
        
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"
        assert DeviceType.REMOTE.value == "remote"


class TestAPIServer:
    """API 服务器测试"""
    
    def test_chat_completion_request_model(self):
        """测试请求模型"""
        from kcake.api import ChatCompletionRequest, Message
        
        request = ChatCompletionRequest(
            model="test_model",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
            temperature=0.7,
            max_tokens=100,
        )
        
        assert request.model == "test_model"
        assert len(request.messages) == 2
        assert request.temperature == 0.7
    
    def test_ollama_generate_request(self):
        """测试 Ollama 请求模型"""
        from kcake.api import OllamaGenerateRequest
        
        request = OllamaGenerateRequest(
            model="llama3.1",
            prompt="Tell me a joke",
            stream=False,
        )
        
        assert request.model == "llama3.1"
        assert request.stream is False


class TestDeviceManager:
    """设备管理器测试"""
    
    def test_backend_enum(self):
        """测试后端枚举"""
        from kcake.devices import Backend
        
        assert Backend.CPU.value == "cpu"
        assert Backend.CUDA.value == "cuda"
        assert Backend.MPS.value == "mps"
    
    def test_backend_priority(self):
        """测试后端优先级"""
        from kcake.devices import DeviceManager, Backend
        
        manager = DeviceManager()
        
        # TPU 应该比 CPU 优先级高
        tpu_priority = manager.get_backend_priority(Backend.TPU)
        cpu_priority = manager.get_backend_priority(Backend.CPU)
        
        assert tpu_priority > cpu_priority


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
