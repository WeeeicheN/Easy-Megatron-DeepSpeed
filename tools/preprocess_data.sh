# It's okay to define your own tokenizer_type, and modify the corresponding tokenizer.py in megatron/tokenizer/

workdir=$PWD
rootdir="$workdir""/../.."

python preprocess_data.py \
       --input ${rootdir}/data/processed_redpajama/arxiv/arxiv.jsonl \
       --output-prefix arxiv \
       --tokenizer-type SentencePieceTokenizer \
       --tokenizer-model ${rootdir}/tokenizers/arxiv_vs256k_msl20.model \
       --vocab-file ${rootdir}/tokenizers/arxiv_vs256k_msl20.vocab \
       --partitions 1 \
       --workers 5

mv ${PWD}/*.bin ${rootdir}/data/processed_redpajama/arxiv/
mv ${PWD}/*.idx ${rootdir}/data/processed_redpajama/arxiv/
