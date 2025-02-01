OUTPUT_DIR=output/
python -m src.relation_encoders.compute_features \
    --dataset scanrefer \
    --output $OUTPUT_DIR \
    --label pred
