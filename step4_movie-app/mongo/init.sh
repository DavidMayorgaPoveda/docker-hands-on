#!/bin/bash
set -e

echo "Importing initial movie data..."

mongoimport \
  --db movies_db \
  --collection movies \
  --file /docker-entrypoint-initdb.d/init.json \
  --jsonArray

echo "Movie data imported successfully."
