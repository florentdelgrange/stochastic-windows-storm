#!/usr/bin/env bash
dot -Tpdf mdp.dot -o mdp.pdf
MP_FILES=`find . -maxdepth 1 -name "*Unfolding*.dot"`
for file in $MP_FILES
do
    dot -Tpdf $file -o ${file%".dot"}.pdf
done
open mdp.pdf *Unfolding*.pdf
