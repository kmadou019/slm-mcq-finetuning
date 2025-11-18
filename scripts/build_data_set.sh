#!/usr/bin/env bash

cd ../data/lisa_sheets
echo "folder,id,content_raw,rubric"
for dir in $(ls);
do
    if test -d "${dir}"; then
        cd $dir
        for sheet in $(ls);
        do
            rubrique=$(grep "Rubrique" $sheet)
            rubrique=${rubrique:10}
            echo $dir","${sheet::12}","\""$(cat $sheet)"\"","\"$rubrique\"
            #sed -i 's/"/""/g' "$sheet"
        done
        cd ..
    fi

done