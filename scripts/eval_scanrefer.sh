export OPENAI_API_KEY="xxxx"
python -m src.eval.eval_scanrefer \
    --features_path output/scanrefer_features_per_scene.pth \
    --top_k 5 \
    --threshold 0.1 \
    --use_vlm
