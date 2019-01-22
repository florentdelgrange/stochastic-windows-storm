dot -Tpdf mdp.dot -o mdp.pdf
MP_FILES=`find . -maxdepth 1 -name "*_ECUnfolding_*.dot"`
for file in $MP_FILES
do
    dot -Tpdf $file -o ${file%".dot"}.pdf
done
open mdp.pdf *_ECUnfolding_*.pdf
