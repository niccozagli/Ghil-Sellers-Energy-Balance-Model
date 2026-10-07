#!/bin/zsh
# Copy each archive to local staging, scan it, then delete the local copy.
S=$1; STAGE=$S/stage; SRC=/Volumes/Nicco/Plasim/extracted
cd /Users/niccolo/Desktop/Ghil_Sellers_Energy_Balance_Model
for dir in $SRC/CONTROL_*; do
  name=${dir:t}
  [[ -e $S/integrity_$name.json ]] && continue
  rm -rf $STAGE; mkdir -p $STAGE/$name
  start=$(date +%s)
  cp $dir/*.nc $STAGE/$name/ || { echo "$name: copy failed"; continue; }
  echo "$name: copied in $(( $(date +%s) - start )) s"
  PYTHONPATH=src uv run python $S/integrity.py $S $STAGE 2>&1 | grep -v Warning
  rm -rf $STAGE
done
echo ALL DONE
