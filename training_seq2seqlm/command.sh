# BART-base/large T5-base/large/3b
sh training_script/single_gpu.sh 5e-5 4 32 1 32 facebook/bart-base LM AdamW /your/output/dir /your/data/dir
sh training_script/single_gpu.sh 5e-5 4 32 1 32 facebook/bart-large LM AdamW /your/output/dir /your/data/dir
sh training_script/single_gpu.sh 5e-4 4 64 1 32 t5-base sentinel Adafactor /your/output/dir /your/data/dir
sh training_script/single_gpu.sh 5e-5 4 32 1 32 t5-large sentinel Adafactor /your/output/dir /your/data/dir
sh training_script/multi_gpu.sh 5e-5 4 4 8 4 t5-3b 2 sentinel Adafactor /your/output/dir /your/data/dir
