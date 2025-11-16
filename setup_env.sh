#!/bin/bash
# ================================
# 自动创建 Python 虚拟环境并安装依赖
# 不需要 sudo
# ================================

# 1. 确认目录在项目根目录
cd ~/heurigym/heurigym || exit 1

# 2. 检查 Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ 没找到 python3，请联系管理员安装"
    exit 1
fi

# 3. 安装 pip（用户级）
echo "➡️ 安装 pip ..."
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --user

# 4. 安装 virtualenv（用户级）
echo "➡️ 安装 virtualenv ..."
~/.local/bin/pip3 install --user virtualenv

# 5. 创建虚拟环境
echo "➡️ 创建虚拟环境 .venv ..."
python3 -m virtualenv .venv

# 6. 激活环境并安装依赖
echo "➡️ 激活虚拟环境并安装依赖 ..."
source .venv/bin/activate
pip install -r requirements.txt

echo "✅ 虚拟环境准备完成！"
echo "以后使用时，先运行："
echo "    cd ~/heurigym/heurigym"
echo "    source .venv/bin/activate"
