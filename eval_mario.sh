#!/bin/bash

# Script to easily run Mario model evaluation in different modes
# Usage: ./eval_mario.sh [--model MODEL_PATH] [--vecnorm VECNORM_PATH] [--episodes N] [--video] [--help]

# Default values
MODEL_PATH="checkpoints_ppo_sb3/2025-03-27T22-18-07/mario_model_2050000_steps.zip"
VECNORM_PATH="checkpoints_ppo_sb3/2025-03-27T22-18-07/mario_model_vecnormalize_2050000_steps.pkl"
EPISODES=3
CAPTURE_VIDEO=false
DEVICE="auto"
VIDEO_FOLDER="./videos"

# Help function
function show_help {
  echo "Usage: ./eval_mario.sh [options]"
  echo "Options:"
  echo "  --model PATH      Path to the model file (default: latest model)"
  echo "  --vecnorm PATH    Path to the VecNormalize statistics file (default: latest stats)"
  echo "  --episodes N      Number of episodes to evaluate (default: 3)"
  echo "  --video           Record videos of the gameplay (default: false)"
  echo "  --device DEVICE   Device to use for inference (cpu, cuda, auto) (default: auto)"
  echo "  --help            Show this help message"
  exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --vecnorm)
      VECNORM_PATH="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --video)
      CAPTURE_VIDEO=true
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      ;;
  esac
done

# Construct video flag
VIDEO_FLAG=""
if [ "$CAPTURE_VIDEO" = true ]; then
  VIDEO_FLAG="--capture-video"
fi

echo "Evaluating Mario model..."
echo "Model path: $MODEL_PATH"
echo "VecNormalize path: $VECNORM_PATH"
echo "Episodes: $EPISODES"
echo "Capture video: $CAPTURE_VIDEO"
echo "Device: $DEVICE"
echo "Video folder: $VIDEO_FOLDER"
echo ""

# Run the evaluation
python ppo_eval.py \
  --model-path "$MODEL_PATH" \
  --vecnormalize-path "$VECNORM_PATH" \
  --eval-episodes "$EPISODES" \
  --device "$DEVICE" \
  --video-folder "$VIDEO_FOLDER" \
  $VIDEO_FLAG

# Check for result
if [ $? -eq 0 ]; then
  echo "Evaluation completed successfully!"
  if [ "$CAPTURE_VIDEO" = true ]; then
    echo "Videos saved to $VIDEO_FOLDER"
  fi
else
  echo "Evaluation failed with error code $?"
fi 