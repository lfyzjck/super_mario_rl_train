#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${GREEN}Super Mario Bros 并行训练脚本${NC}"
    echo ""
    echo "用法: ./run_parallel_training.sh [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --mode <basic|apex>    训练模式: basic (基本并行) 或 apex (Ape-X DQN)"
    echo "  -p, --processes <num>      并行进程数量 (默认: 4)"
    echo "  -e, --episodes <num>       训练回合数 (默认: 2000)"
    echo "  -b, --batch-size <num>     批量大小 (默认: 64 for basic, 512 for apex)"
    echo "  -r, --render               渲染游戏画面 (可选)"
    echo "  -h, --help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./run_parallel_training.sh --mode basic --processes 4 --episodes 2000"
    echo "  ./run_parallel_training.sh --mode apex --processes 8 --episodes 1000 --batch-size 256"
    echo ""
}

# 默认参数
MODE="basic"
PROCESSES=4
EPISODES=2000
BATCH_SIZE=""
RENDER=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--mode)
            MODE="$2"
            shift
            shift
            ;;
        -p|--processes)
            PROCESSES="$2"
            shift
            shift
            ;;
        -e|--episodes)
            EPISODES="$2"
            shift
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        -r|--render)
            RENDER="--render"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 设置默认批量大小（如果未指定）
if [ -z "$BATCH_SIZE" ]; then
    if [ "$MODE" == "basic" ]; then
        BATCH_SIZE=64
    else
        BATCH_SIZE=512
    fi
fi

# 运行训练
if [ "$MODE" == "basic" ]; then
    echo -e "${YELLOW}启动基本并行训练...${NC}"
    echo "进程数: $PROCESSES, 回合数: $EPISODES, 批量大小: $BATCH_SIZE"
    python parallel_train.py --num-processes $PROCESSES --episodes $EPISODES --batch-size $BATCH_SIZE $RENDER
elif [ "$MODE" == "apex" ]; then
    echo -e "${YELLOW}启动 Ape-X DQN 并行训练...${NC}"
    echo "Actor进程数: $PROCESSES, 回合数: $EPISODES, 批量大小: $BATCH_SIZE"
    python apex_parallel_train.py --num-actors $PROCESSES --episodes $EPISODES --batch-size $BATCH_SIZE $RENDER
else
    echo "错误: 未知模式 '$MODE'. 请使用 'basic' 或 'apex'."
    show_help
    exit 1
fi 