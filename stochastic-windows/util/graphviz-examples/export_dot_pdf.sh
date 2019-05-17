#!/usr/bin/env bash
MP_FILES=`find . -maxdepth 1 -name "*.dot"`
for file in $MP_FILES
do
    dot -Tpdf $file -o ${file%".dot"}.pdf
done
open *.pdf
