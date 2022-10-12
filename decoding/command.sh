# bart-base
sh decoding/single_gpu.sh 2 64 LM /trained/seq2seqlm /trained/coco /your/output/dir /your/data/dir
# bart-large
sh decoding/single_gpu.sh 1 64 LM /trained/seq2seqlm /trained/coco /your/output/dir /your/data/dir
# T5-base
sh decoding/single_gpu.sh 5 64 sentinel /trained/seq2seqlm /trained/coco /your/output/dir /your/data/dir
# T5-large
sh decoding/single_gpu.sh 5e-1 64 sentinel /trained/seq2seqlm /trained/coco /your/output/dir /your/data/dir
# T5-3b
sh decoding/single_gpu.sh 5e-1 8 sentinel /trained/seq2seqlm /trained/coco /your/output/dir /your/data/dir