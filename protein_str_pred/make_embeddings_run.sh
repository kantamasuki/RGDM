for i in {0..1}; do
    CUDA_VISIBLE_DEVICES=$i python make_embeddings.py --splits splits/limit256.csv --reference_only --num_workers 2 --worker_id $i &
done
