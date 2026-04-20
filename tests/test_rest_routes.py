"""
REST路由简单测试 - 践行毛泽东思想：实践是检验真理的唯一标准
"""

import pytest
from src.api.rest_routes import router, setup_routes


class TestRouter:
    """路由测试"""
    
    def test_router_exists(self):
        """测试router存在"""
        assert router is not None
    
    def test_setup_routes_function_exists(self):
        """测试setup_routes函数存在"""
        assert setup_routes is not None
        assert callable(setup_routes)


class TestRouteSetup:
    """路由设置测试"""
    
    def test_setup_routes_with_mock(self):
        """测试setup_routes调用不报错"""
        # 使用mock对象调用setup_routes
        mock_app = type('MockApp', (), {})()
        mock_engine = type('MockEngine', (), {'clear_cache': lambda: None, 'get_model_info': lambda: {}})()
        mock_cluster = None
        
        # 这个调用不应该报错
        setup_routes(mock_app, mock_engine, mock_cluster)
    
    def test_setup_routes_with_cluster_manager(self):
        """测试带集群管理器的setup_routes调用"""
        mock_app = type('MockApp', (), {})()
        mock_engine = type('MockEngine', (), {'clear_cache': lambda: None, 'get_model_info': lambda: {'is_loaded': True, 'model_name': 'test'}})()
        mock_cluster = type('MockCluster', (), {'get_cluster_status': lambda: {'nodes': 1}})()
        
        # 这个调用不应该报错
        setup_routes(mock_app, mock_engine, mock_cluster)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])