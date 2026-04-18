#!/bin/bash
# KCake 测试运行脚本

set -e

echo "========================================"
echo "KCake 测试运行脚本"
echo "========================================"

# 进入项目目录
cd /root/workspace/kcake

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import pytest; print(f'pytest: {pytest.__version__}')" || echo "pytest未安装"

# 运行测试
echo ""
echo "运行单元测试..."
python3 -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing || true

echo ""
echo "测试完成!"
