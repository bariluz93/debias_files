#!/bin/bash
set -e

good="False"
depth=0
while [ "$good" != "True" ]
do
   if [ $depth == 10 ]; then
     good="True"
   else
    sh /cs/labs/gabis/bareluz/nematus_clean/debias_files/flow_scripts/run_all_configurations_locations.sh
    last_dir=$(ls -td /cs/usr/bareluz/gabi_labs/nematus_clean/debias_outputs/results/*/ | head -1)
    echo "last_dir ${last_dir}"
    good=`python /cs/labs/gabis/bareluz/nematus_clean/debias_files/test_file.py ${last_dir}`
    echo "good ${good}"
    ((depth=depth+1))
    echo "depth depth depth depth depth depth depth depth depth depth depth depth depth depth depth ${depth}"
   fi
done

