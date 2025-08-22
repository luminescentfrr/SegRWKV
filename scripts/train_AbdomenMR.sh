RWKV_MODEL=$1
PRED_OUTPUT_PATH="data/nnUNet_results/Dataset702_AbdomenMR/${RWKV_MODEL}__nnUNetPlans__2d/pred_results"
EVAL_METRIC_PATH="data/nnUNet_results/Dataset702_AbdomenMR/${RWKV_MODEL}__nnUNetPlans__2d"
GPU_ID="0,1"

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_DEBUG=INFO
export PYTHONWARNINGS="ignore"

# train with proper environment setup
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 702 2d all -tr ${RWKV_MODEL} -num_gpus 2 &&

echo "Predicting..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "data/nnUNet_raw/Dataset702_AbdomenMR/imagesTs" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 702 \
    -c 2d \
    -tr "${RWKV_MODEL}" \
    --disable_tta \
    -f all \
    -chk "checkpoint_249.pth" &&

echo "Computing dice..."
python evaluation/abdomen_DSC_Eval.py \
    --gt_path "data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_DSC.csv"  &&

echo "Computing NSD..."
python evaluation/abdomen_NSD_Eval.py \
    --gt_path "data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_NSD.csv" &&

echo "Done."