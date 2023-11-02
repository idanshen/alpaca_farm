$run=$1
$classification_label_key=$2
$gpu=$3

CUDA_VISIBLE_DEVICES=$gpu python3 \
    ./examples/seahorse_classification.py \
    --output_dir "./$run" \ 
    --run_name  "$run" \
    --weight_decay 0.02 \
    --classification_label_key $classification_label_key