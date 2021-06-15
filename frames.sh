#!/bin/bash

cd labeled

for vid in {0..4}
do
  if [[ ! -e $vid ]]; then
      mkdir -p $vid
  else
      echo "$vid already exists " 1>&2
      exit 1
  fi
  ffmpeg -i "$vid".hevc "$vid"/%04d.jpg
done
