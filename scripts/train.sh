export XFL_CONFIG=experiments/experiments.yaml

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch --config_file accelerate.yaml  --debug -m src.train.train_join
