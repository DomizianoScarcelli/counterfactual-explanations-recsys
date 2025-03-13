echo "\n-------UNTARGETED CATEGORIZED-----\n"
sh scripts/evaluation.sh 1 1 BERT4Rec untargeted categorized
echo "\n-------UNTARGETED UNCATEGORIZED-----\n"
sh scripts/evaluation.sh 1 1 BERT4Rec untargeted uncategorized
echo "\n-------TARGETED CATEGORIZED-----\n"
sh scripts/evaluation.sh 1 6 BERT4Rec targeted categorized 
echo "\n-------TARGETED UNCATEGORIZED-----\n"
sh scripts/evaluation.sh 1 4 BERT4Rec targeted uncategorized

echo "-------UNTARGETED UNCATEGORIZED (GRU4Rec)-----"
sh scripts/evaluation.sh 1 1 GRU4Rec untargeted 
