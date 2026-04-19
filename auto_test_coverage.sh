#!/bin/bash
# 24小时不间断开发 - 自动化测试覆盖率脚本
# 践行毛泽东思想：实践是检验真理的唯一标准

echo "========================================"
echo "AI-OA项目自动化测试覆盖率工具"
echo "时间: $(date)"
echo "践行毛泽东思想: 实事求是，每日有进步"
echo "========================================"

# 创建测试覆盖率目录
mkdir -p coverage_reports

# 第一步: 检查Python环境
echo "1. 检查Python环境..."
python3 --version
pip list | grep -E "pytest|cov|torch"

# 第二步: 运行单元测试
echo ""
echo "2. 运行单元测试..."
cd /root/workspace/kcake
python3 -m pytest tests/ -v --tb=short 2>&1 | tee coverage_reports/unit_tests_$(date +%Y%m%d_%H%M).log

# 第三步: 生成覆盖率报告
echo ""
echo "3. 生成测试覆盖率报告..."
python3 -m pytest tests/ --cov=src --cov-report=html:coverage_reports/html --cov-report=term 2>&1 | tee coverage_reports/coverage_$(date +%Y%m%d_%H%M).log

# 第四步: 检查覆盖率是否达标
echo ""
echo "4. 检查覆盖率是否达标..."
COVERAGE=$(python3 -m pytest tests/ --cov=src --cov-report=term 2>&1 | grep "TOTAL" | awk '{print $4}' | sed 's/%//')
echo "当前测试覆盖率: ${COVERAGE}%"

# 第五步: 根据覆盖率决定下一步行动
if (( $(echo "$COVERAGE < 95" | bc -l) )); then
    echo "⚠️ 覆盖率未达95%标准，需要补充测试"
    echo "需要补充测试的模块:"
    python3 -m pytest tests/ --cov=src --cov-report=term-missing 2>&1 | grep "Missing" | head -10
else
    echo "✅ 测试覆盖率达标 (≥95%)"
fi

# 第六步: 自动提交代码到GitHub
echo ""
echo "5. 自动提交代码到GitHub..."
git add .
git commit -m "auto-test: $(date +%Y-%m-%d_%H:%M) - 覆盖率${COVERAGE}%" || echo "无新更改"
git push origin master || echo "推送失败，检查网络"

# 第七步: 生成开发报告
echo ""
echo "6. 生成开发进度报告..."
cat > coverage_reports/progress_report_$(date +%Y%m%d_%H%M).md << EOF
# AI-OA项目开发进度报告
## 测试覆盖率检查 - $(date)

### 执行时间
$(date)

### 测试覆盖率
- **当前覆盖率**: ${COVERAGE}%
- **目标覆盖率**: 95%
- **状态**: $([ $(echo "$COVERAGE >= 95" | bc -l) -eq 1 ] && echo "✅ 达标" || echo "⚠️ 未达标")

### 测试执行情况
- 单元测试: 已执行
- 集成测试: 待完善
- API测试: 待完善

### 下一步行动
$([ $(echo "$COVERAGE >= 95" | bc -l) -eq 1 ] && echo "1. 开始完善集成测试" || echo "1. 补充单元测试达到95%覆盖率")

### 践行毛泽东思想
- 实事求是: 真实反映测试覆盖率
- 实践论: 通过测试验证代码质量
- 每日有进步: 每次测试都有提升

EOF

echo ""
echo "========================================"
echo "测试覆盖率检查完成"
echo "详细报告: coverage_reports/progress_report_$(date +%Y%m%d_%H%M).md"
echo "HTML报告: coverage_reports/html/index.html"
echo "========================================"