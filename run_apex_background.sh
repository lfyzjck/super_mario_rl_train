#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数 - 针对低内存环境优化
NUM_ACTORS=2
BUFFER_SIZE=50000
BATCH_SIZE=64
EPISODES=1000
LOG_FILE="apex_training.log"
CHECKPOINT_DIR="checkpoints/$(date +%Y-%m-%dT%H-%M-%S)"

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
        --log)
            LOG_FILE="$2"
            shift
            shift
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift
            shift
            ;;
        --help)
            echo "用法: ./run_apex_background.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --actors <num>         Actor进程数量 (默认: 2)"
            echo "  --buffer-size <num>    缓冲区大小 (默认: 50000)"
            echo "  --batch-size <num>     批量大小 (默认: 64)"
            echo "  --episodes <num>       训练回合数 (默认: 1000)"
            echo "  --log <file>           日志文件路径 (默认: apex_training.log)"
            echo "  --checkpoint-dir <dir> 检查点保存目录 (默认: checkpoints/日期时间)"
            echo "  --help                 显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 创建检查点目录
mkdir -p "$CHECKPOINT_DIR"

# 显示训练配置
echo -e "${GREEN}Super Mario Bros Ape-X DQN 后台训练${NC}"
echo "训练配置:"
echo "  Actor进程数: $NUM_ACTORS"
echo "  缓冲区大小: $BUFFER_SIZE"
echo "  批量大小: $BATCH_SIZE"
echo "  训练回合数: $EPISODES"
echo "  日志文件: $LOG_FILE"
echo "  检查点目录: $CHECKPOINT_DIR"
echo ""
echo "训练将在后台运行，日志将写入 $LOG_FILE"
echo "可以使用 'tail -f $LOG_FILE' 查看训练进度"
echo ""

# 安装必要的依赖
if ! python -c "import psutil" &> /dev/null; then
    echo -e "${YELLOW}安装 psutil 以监控内存使用...${NC}"
    pip install psutil
fi

# 启动训练进程
echo -e "${YELLOW}启动 Ape-X DQN 训练...${NC}"
nohup python apex_parallel_train.py \
    --num-actors $NUM_ACTORS \
    --buffer-size $BUFFER_SIZE \
    --batch-size $BATCH_SIZE \
    --episodes $EPISODES \
    > "$LOG_FILE" 2>&1 &

# 获取进程ID
TRAIN_PID=$!
echo "训练进程已启动，PID: $TRAIN_PID"
echo "进程ID已保存到 $CHECKPOINT_DIR/pid.txt"
echo $TRAIN_PID > "$CHECKPOINT_DIR/pid.txt"

# 创建监控脚本
cat > "$CHECKPOINT_DIR/monitor.sh" << EOF
#!/bin/bash
if ps -p $TRAIN_PID > /dev/null; then
    echo "训练进程 (PID: $TRAIN_PID) 正在运行"
    echo "最近的日志:"
    tail -n 20 "$LOG_FILE"
    
    # 显示内存使用情况
    if command -v ps &> /dev/null; then
        echo ""
        echo "内存使用情况:"
        ps -o pid,ppid,%cpu,%mem,rss,command -p $TRAIN_PID
    fi
else
    echo "训练进程 (PID: $TRAIN_PID) 已结束"
    echo "查看完整日志: $LOG_FILE"
fi
EOF

chmod +x "$CHECKPOINT_DIR/monitor.sh"
echo "监控脚本已创建: $CHECKPOINT_DIR/monitor.sh"

# 创建停止脚本
cat > "$CHECKPOINT_DIR/stop.sh" << EOF
#!/bin/bash
if ps -p $TRAIN_PID > /dev/null; then
    echo "正在停止训练进程 (PID: $TRAIN_PID)..."
    kill $TRAIN_PID
    sleep 2
    if ps -p $TRAIN_PID > /dev/null; then
        echo "进程未响应，强制终止..."
        kill -9 $TRAIN_PID
    fi
    echo "训练进程已停止"
else
    echo "训练进程 (PID: $TRAIN_PID) 已不存在"
fi
EOF

chmod +x "$CHECKPOINT_DIR/stop.sh"
echo "停止脚本已创建: $CHECKPOINT_DIR/stop.sh"

echo -e "${GREEN}训练已在后台启动${NC}"
echo "使用以下命令查看训练状态:"
echo "  $CHECKPOINT_DIR/monitor.sh"
echo "使用以下命令停止训练:"
echo "  $CHECKPOINT_DIR/stop.sh" 