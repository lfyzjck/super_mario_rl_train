#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Super Mario Bros Ape-X DQN 低内存训练${NC}"
echo ""
echo "此脚本使用优化的参数在低内存环境中运行 Ape-X DQN 训练"
echo ""

# 默认参数 - 针对低内存环境优化
NUM_ACTORS=2
BUFFER_SIZE=50000
BATCH_SIZE=64
EPISODES=1000

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --actors)
            NUM_ACTORS="$2"
            shift
            shift
            ;;
        --buffer-size)
            BUFFER_SIZE="$2"
            shift
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --episodes)
            EPISODES="$2"
            shift
            shift
            ;;
        --render)
            RENDER="--render"
            shift
            ;;
        --help)
            echo "用法: ./run_apex_low_memory.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --actors <num>       Actor进程数量 (默认: 2)"
            echo "  --buffer-size <num>  缓冲区大小 (默认: 50000)"
            echo "  --batch-size <num>   批量大小 (默认: 64)"
            echo "  --episodes <num>     训练回合数 (默认: 1000)"
            echo "  --render             渲染游戏画面"
            echo "  --help               显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 安装内存监控工具（如果需要）
if ! python -c "import psutil" &> /dev/null; then
    echo -e "${YELLOW}安装 psutil 以监控内存使用...${NC}"
    pip install psutil
fi

# 显示训练配置
echo -e "${YELLOW}训练配置:${NC}"
echo "Actor进程数: $NUM_ACTORS"
echo "缓冲区大小: $BUFFER_SIZE"
echo "批量大小: $BATCH_SIZE"
echo "训练回合数: $EPISODES"
if [ ! -z "$RENDER" ]; then
    echo "渲染: 启用"
else
    echo "渲染: 禁用"
fi
echo ""

# 确认继续
read -p "是否继续训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消训练"
    exit 0
fi

# 运行训练
echo -e "${YELLOW}启动 Ape-X DQN 训练...${NC}"
python apex_parallel_train.py \
    --num-actors $NUM_ACTORS \
    --buffer-size $BUFFER_SIZE \
    --batch-size $BATCH_SIZE \
    --episodes $EPISODES \
    $RENDER

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
    echo -e "${GREEN}训练成功完成!${NC}"
else
    echo -e "${YELLOW}训练异常终止，退出代码: $?${NC}"
fi 