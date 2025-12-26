#!/bin/bash
# Run all experiments for AlexNet iFood2019 project

echo "=============================================="
echo "AlexNet iFood2019 - Run All Experiments"
echo "=============================================="

# Configuration
DATA_DIR=${DATA_DIR:-"./data"}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-0.01}
SAVE_DIR=${SAVE_DIR:-"./checkpoints"}
EVAL_DIR=${EVAL_DIR:-"./evaluation_results"}
ANALYSIS_DIR=${ANALYSIS_DIR:-"./analysis_results"}

# Models to train
MODELS=(
    "alexnet_baseline"
    "alexnet_mod1"
    "alexnet_mod2"
    "alexnet_combined"
)

# Create directories
mkdir -p $SAVE_DIR
mkdir -p $EVAL_DIR
mkdir -p $ANALYSIS_DIR

# Train all models
echo ""
echo "Training ${#MODELS[@]} models..."
echo ""

for model in "${MODELS[@]}"; do
    echo "=============================================="
    echo "Training: $model"
    echo "=============================================="
    
    python src/train.py \
        --data_dir $DATA_DIR \
        --model_name $model \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --save_dir $SAVE_DIR
    
    echo "✓ Completed: $model"
    echo ""
done

# Evaluate all models
echo ""
echo "Evaluating all models..."
echo ""

for model in "${MODELS[@]}"; do
    echo "Evaluating: $model"
    
    python src/evaluate.py \
        --data_dir $DATA_DIR \
        --model_path $SAVE_DIR/${model}_best.pth \
        --model_name $model \
        --split val \
        --output_dir $EVAL_DIR
    
    echo "✓ Evaluated: $model"
done

# Run comparative analysis
echo ""
echo "Running comparative analysis..."

python src/analysis.py \
    --checkpoint_dir $SAVE_DIR \
    --eval_dir $EVAL_DIR \
    --output_dir $ANALYSIS_DIR \
    --models ${MODELS[@]}

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  Checkpoints: $SAVE_DIR"
echo "  Evaluation: $EVAL_DIR"
echo "  Analysis: $ANALYSIS_DIR"
