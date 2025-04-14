#!/bin/bash

# Define paths
CKPT_PATH="your/path/to/fuzhouhua/outputs/whisper-large-v3/checkpoint-12000"
PRETRAINED_PATH="your/path/to/fuzhouhua/models/openai/whisper-large-v3"
RELEASE_PATH="your/path/to/fuzhouhua/releases/whisper-large-v3"

# Create the target directory if it doesn't exist
mkdir -p $RELEASE_PATH

# Copy necessary files from the fine-tuned model
cp $CKPT_PATH/config.json $RELEASE_PATH/
cp $CKPT_PATH/generation_config.json $RELEASE_PATH/
cp $CKPT_PATH/model-00001-of-00002.safetensors $RELEASE_PATH/
cp $CKPT_PATH/model-00002-of-00002.safetensors $RELEASE_PATH/
cp $CKPT_PATH/model.safetensors.index.json $RELEASE_PATH/
cp $CKPT_PATH/preprocessor_config.json $RELEASE_PATH/

# Copy tokenizer and configuration files from the original pretrained model
cp $PRETRAINED_PATH/added_tokens.json $RELEASE_PATH/
cp $PRETRAINED_PATH/merges.txt $RELEASE_PATH/
cp $PRETRAINED_PATH/normalizer.json $RELEASE_PATH/
cp $PRETRAINED_PATH/special_tokens_map.json $RELEASE_PATH/
cp $PRETRAINED_PATH/tokenizer.json $RELEASE_PATH/
cp $PRETRAINED_PATH/tokenizer_config.json $RELEASE_PATH/
cp $PRETRAINED_PATH/vocab.json $RELEASE_PATH/

echo "Files copied to $RELEASE_PATH"

cp -r $RELEASE_PATH /data/oss_bucket_0
echo "Release directory copied to /data/oss_bucket_0"