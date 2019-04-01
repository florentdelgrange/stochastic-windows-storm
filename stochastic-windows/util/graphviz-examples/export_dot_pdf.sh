#!/usr/bin/env bash
#dot -Tpdf mdp.dot -o mdp.pdf
#dot -Tpdf dfwMemoryExample.dot -o dfwMemoryExample.pdf
#MP_FILES=`find . -maxdepth 1 -name "*Unfolding*.dot"`
MP_FILES=`find . -maxdepth 1 -name "*.dot"`
for file in $MP_FILES
do
    dot -Tpdf $file -o ${file%".dot"}.pdf
done
#open mdp.pdf *Unfolding*.pdf
open *.pdf
