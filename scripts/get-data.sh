#!/usr/bin/env bash
# Download the three public datasets at pinned revisions and check their SHA-256.
#   Civil Comments (CC0 1.0)   huggingface.co/datasets/google/civil_comments
#   HateCheck      (CC BY 4.0) huggingface.co/datasets/Paul/hatecheck  (Röttger et al., ACL 2021)
#   LDNOOBW English word list (CC BY 4.0) github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words
set -euo pipefail
cd "$(dirname "$0")/../data/raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data/raw"; cd "$(dirname "$0")/../data/raw"; }

CIVIL=https://huggingface.co/datasets/google/civil_comments/resolve/f2970eb3a55777454c94069077cc8d9b5866312d/data
HC=https://huggingface.co/datasets/Paul/hatecheck/resolve/9d2ac89df04254e5c427bcc8d61b6d6c83a1f59b/test.csv
WORDS=https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/5faf2ba42d7b1c0977169ec3611df25a3c08eb13/en

get() { [ -f "$1" ] || { echo "downloading $1"; curl -fsSL -o "$1" "$2"; }; }
# only the first of the two training shards: 150k training comments are sampled from it
get civil_train-00000-of-00002.parquet "$CIVIL/train-00000-of-00002.parquet"
get civil_validation-00000-of-00001.parquet "$CIVIL/validation-00000-of-00001.parquet"
get civil_test-00000-of-00001.parquet "$CIVIL/test-00000-of-00001.parquet"
get hatecheck.csv "$HC"
get ldnoobw_en.txt "$WORDS"

shasum -a 256 -c <<'SUMS'
410f803ff161e7c4ba005db7de5b444cac413f4861bf056b0fd69d670d75aa08  civil_test-00000-of-00001.parquet
c20f01c3aecbdd942886cacb0ee67e995df33bc712a622fde75927ed3d6ccefe  civil_train-00000-of-00002.parquet
2e0eb65474e7e1290df8689fc93eea783160c898c09ae15382c0a9662367bd03  civil_validation-00000-of-00001.parquet
ef1c1b95f17c1289a4fa42aefdd7b22ec7758d9420539cf751cdf4fc0854d83b  hatecheck.csv
af851ecef1d5f212caba17339b12ac39cc2fef7d78c74876f67237644fcee8bd  ldnoobw_en.txt
SUMS
