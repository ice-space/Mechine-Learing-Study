#!/bin/bash

# 获取所有文件的哈希值
git rev-list --objects --all | sort -k 2 > allfileshas.txt

# 获取文件大小并排序
cat allfileshas.txt | cut -d' ' -f1 | while read -r sha; do
  size=$(git cat-file -s "$sha")
  path=$(git ls-tree -r --name-only HEAD | grep "$sha")
  if [[ -n $path ]]; then
    echo "$size $path"
  fi
done | sort -nr | head -n 50
