export OPENAI_API_KEY="xxxx"
python -m src.eval.eval_nr3d \
    --features_path output/nr3d_features_per_scene_pred_label.pth \
    --top_k 5 \
    --threshold 0 \
    --label_type gt \
    --use_vlm 
