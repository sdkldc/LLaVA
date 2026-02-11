#!/bin/bash

echo "========================================="
echo "Quick Checkpoint Change Verification"
echo "========================================="
echo ""

BASE="/home/deokhyeon/Documents/LLaVA/checkpoints/llava-v1.5-7b-token32-batch128"

for ckpt in checkpoint-980 checkpoint-990 checkpoint-1000; do
    echo "📁 $ckpt"
    echo "   Default adapter:"
    if [ -f "$BASE/$ckpt/adapter_model.safetensors" ]; then
        SIZE=$(stat -c%s "$BASE/$ckpt/adapter_model.safetensors")
        HASH=$(md5sum "$BASE/$ckpt/adapter_model.safetensors" | cut -d' ' -f1)
        echo "     Size: $SIZE bytes"
        echo "     MD5:  $HASH"
    else
        echo "     ❌ File not found"
    fi
    
    echo "   Summary_utilizer adapter:"
    if [ -f "$BASE/$ckpt/summary_utilizer/adapter_model.safetensors" ]; then
        SIZE=$(stat -c%s "$BASE/$ckpt/summary_utilizer/adapter_model.safetensors")
        HASH=$(md5sum "$BASE/$ckpt/summary_utilizer/adapter_model.safetensors" | cut -d' ' -f1)
        echo "     Size: $SIZE bytes"
        echo "     MD5:  $HASH"
    else
        echo "     ❌ File not found"
    fi
    echo ""
done

echo ""
echo "========================================="
echo "Hash Comparison Summary"
echo "========================================="
echo "If all MD5 hashes are DIFFERENT, weights are changing! ✅"
echo "If hashes are SAME, weights are NOT changing! ❌"
