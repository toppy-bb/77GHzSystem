#!/bin/sh

if [ $# -ne 2 ]; then
  echo "Usage: ./pre.sh data_file sampling_rate"
  exit 1;
else
  if [ -f $1 ]; then
    python3 code/AmPh_Scatter.py $1 $2
    #python3 code/PS.py
  else
    echo "$0: $1: No such file or directory"
    exit 1;
  fi
fi



