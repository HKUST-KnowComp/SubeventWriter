# All-at-once Seq2Seq: bart-base/large T5-base/large/3b
sh baseline/single_gpu.sh 5e-5 4 32 1 64 facebook/bart-base LM AdamW /your/output/dir /your/data/dir
sh baseline/single_gpu.sh 1e-5 4 32 1 64 facebook/bart-large LM AdamW /your/output/dir /your/data/dir
sh baseline/single_gpu.sh 1e-3 4 32 1 64 t5-base sentinel Adafactor /your/output/dir /your/data/dir
sh baseline/single_gpu.sh 5e-4 4 32 1 32 t5-large sentinel Adafactor /your/output/dir /your/data/dir
sh baseline/multi_gpu.sh 1e-4 4 4 4 4 t5-3b 2 sentinel Adafactor /your/output/dir /your/data/dir

# Zero-shot Large LM: GPT-J T5-11b
sh baseline/run_gpt-j.sh 4 4 /your/output/dir /your/data/dir
sh baseline/run_t5-11b.sh 2 /your/output/dir /your/data/dir

# Top-one Similar Sequence: Glove, Sentence-BERT
python top1_sim_baseline.py --train_file /train/file/path --valid_file /valid/file/path --test_file /test/file/path \
--output_dir /your/output/path --sim_func glove --embedding_path /path/to/glove.6B.300d.txt

python top1_sim_baseline.py --train_file /train/file/path --valid_file /valid/file/path --test_file /test/file/path \
--output_dir /your/output/path --sim_func sbert --model_name_or_path sentence-transformers/bert-base-wikipedia-sections-mean-tokens
