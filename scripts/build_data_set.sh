#!/usr/bin/env bash

cd ../data/lisa_sheets
echo "id,item"
for dir in $(ls);
do
    if test -d "${dir}"; then
        cd $dir
        for sheet in $(ls);
        do
            item=$(grep "Item_parent_short" $sheet)
            item=${item:19}
            echo ${sheet::7}","\"$item\"
            #sed -i 's/"/""/g' "$sheet"
            break
        done
        cd ..
    fi

done