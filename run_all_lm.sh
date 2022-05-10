SEED=$1

echo "seed: ${SEED}"

# Combined main setting

#sh lm.sh dm 10 3 0 0 0 1 0 0 0 $SEED
#sh lm.sh psd 10 3 0 0 0 1 0 0 0 $SEED
#sh lm.sh eds 10 3 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptg 10 3 0 0 0 1 0 0 0 $SEED
#sh lm.sh ud 10 3 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptb-phrase 10 3 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptb-pos 10 3 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptb-func 10 3 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptb-all 10 3 0 0 0 1 0 0 0 $SEED


# Combined setting with POS inputs

#sh lm.sh dm 10 3 0 0 0 1 0 0 0 $SEED upos mrp/upos.training.json
#sh lm.sh psd 10 3 0 0 0 1 0 0 0 $SEED upos mrp/upos.training.json
#sh lm.sh eds 10 3 0 0 0 1 0 0 0 $SEED upos mrp/upos.training.json
#sh lm.sh ptg 10 3 0 0 0 1 0 0 0 $SEED upos mrp/upos.training.json
#sh lm.sh ud 10 3 0 0 0 1 0 0 0 $SEED upos mrp/upos.training.json
#sh lm.sh ptb-phrase 10 3 0 0 0 1 0 0 0 $SEED upos mrp/upos.training.json
#sh lm.sh ptb-pos 10 3 0 0 0 1 0 0 0 $SEED upos mrp/upos.training.json
#sh lm.sh ptb-func 10 3 0 0 0 1 0 0 0 $SEED upos mrp/upos.training.json
#sh lm.sh ptb-all 10 3 0 0 0 1 0 0 0 $SEED upos mrp/upos.training.json
#sh lm.sh empty 10 3 0 0 0 1 0 0 0 $SEED upos mrp/upos.training.json

# GNN Baselines

#sh lm.sh dm 10 3 0 0 0 1 0 0 0 $SEED gcn
#sh lm.sh psd 10 3 0 0 0 1 0 0 0 $SEED gcn
#sh lm.sh eds 10 3 0 0 0 1 0 0 0 $SEED gcn
#sh lm.sh ptg 10 3 0 0 0 1 0 0 0 $SEED gcn
#sh lm.sh ud 10 3 0 0 0 1 0 0 0 $SEED gcn
#sh lm.sh ptb-phrase 10 3 0 0 0 1 0 0 0 $SEED gcn
#sh lm.sh ptb-pos 10 3 0 0 0 1 0 0 0 $SEED gcn
#sh lm.sh ptb-func 10 3 0 0 0 1 0 0 0 $SEED gcn
#sh lm.sh ptb-all 10 3 0 0 0 1 0 0 0 $SEED gcn

#sh lm.sh dm 10 3 0 0 0 1 0 0 0 $SEED gat
#sh lm.sh psd 10 3 0 0 0 1 0 0 0 $SEED gat
#sh lm.sh eds 10 3 0 0 0 1 0 0 0 $SEED gat
#sh lm.sh ptg 10 3 0 0 0 1 0 0 0 $SEED gat
#sh lm.sh ud 10 3 0 0 0 1 0 0 0 $SEED gat
#sh lm.sh ptb-phrase 10 3 0 0 0 1 0 0 0 $SEED gat
#sh lm.sh ptb-pos 10 3 0 0 0 1 0 0 0 $SEED gat
#sh lm.sh ptb-func 10 3 0 0 0 1 0 0 0 $SEED gat
#sh lm.sh ptb-all 10 3 0 0 0 1 0 0 0 $SEED gat

#sh lm.sh dm 10 3 0 0 0 1 0 0 0 $SEED rgcn
#sh lm.sh psd 10 3 0 0 0 1 0 0 0 $SEED rgcn
#sh lm.sh eds 10 3 0 0 0 1 0 0 0 $SEED rgcn
#sh lm.sh ptg 10 3 0 0 0 1 0 0 0 $SEED rgcn
#sh lm.sh ud 10 3 0 0 0 1 0 0 0 $SEED rgcn
#sh lm.sh ptb-phrase 10 3 0 0 0 1 0 0 0 $SEED rgcn
#sh lm.sh ptb-pos 10 3 0 0 0 1 0 0 0 $SEED rgcn
#sh lm.sh ptb-func 10 3 0 0 0 1 0 0 0 $SEED rgcn
#sh lm.sh ptb-all 10 3 0 0 0 1 0 0 0 $SEED rgcn


# Combined PR

#sh lm.sh dm 10 3 0 0 0 1 1 1 0 $SEED
#sh lm.sh psd 10 3 0 0 0 1 1 1 0 $SEED
#sh lm.sh eds 10 3 0 0 0 1 1 1 0 $SEED
#sh lm.sh ptg 10 3 0 0 0 1 1 1 0 $SEED
#sh lm.sh ud 10 3 0 0 0 1 1 1 0 $SEED
#sh lm.sh ptb-phrase 10 3 0 0 0 1 1 1 0 $SEED
#sh lm.sh ptb-pos 10 3 0 0 0 1 1 1 0 $SEED
#sh lm.sh ptb-func 10 3 0 0 0 1 1 1 0 $SEED
#sh lm.sh ptb-all 10 3 0 0 0 1 1 1 0 $SEED


# LM, graph

#sh lm.sh dm 10 2 0 0 0 1 0 0 0 $SEED
#sh lm.sh psd 10 2 0 0 0 1 0 0 0 $SEED
#sh lm.sh eds 10 2 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptg 10 2 0 0 0 1 0 0 0 $SEED
#sh lm.sh ud 10 2 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptb 10 2 0 0 0 1 0 0 0 $SEED


# Combined, no graph tokens

#sh lm.sh dm 10 3 0 0 0 1 0 0 1 $SEED
#sh lm.sh psd 10 3 0 0 0 1 0 0 1 $SEED
#sh lm.sh eds 10 3 0 0 0 1 0 0 1 $SEED
#sh lm.sh ptg 10 3 0 0 0 1 0 0 1 $SEED
#sh lm.sh ud 10 3 0 0 0 1 0 0 1 $SEED
#sh lm.sh ptb-phrase 10 3 0 0 0 1 0 0 1 $SEED                                  
#sh lm.sh ptb-pos 10 3 0 0 0 1 0 0 1 $SEED
#sh lm.sh ptb-func 10 3 0 0 0 1 0 0 1 $SEED
#sh lm.sh ptb-all 10 3 0 0 0 1 0 0 1 $SEED


# Combined, shuffle ablations

#sh lm.sh dm 10 3 0 1 0 1 0 0 0 $SEED
#sh lm.sh psd 10 3 0 1 0 1 0 0 0 $SEED
#sh lm.sh eds 10 3 0 1 0 1 0 0 0 $SEED
#sh lm.sh ptg 10 3 0 1 0 1 0 0 0 $SEED
#sh lm.sh ud 10 3 0 1 0 1 0 0 0 $SEED
#sh lm.sh ptb-phrase 10 3 0 1 0 1 0 0 0 $SEED                                  
#sh lm.sh ptb-pos 10 3 0 1 0 1 0 0 0 $SEED
#sh lm.sh ptb-func 10 3 0 1 0 1 0 0 0 $SEED
#sh lm.sh ptb-all 10 3 0 1 0 1 0 0 0 $SEED

#sh lm.sh dm 10 3 0 0 1 0 0 0 0 $SEED
#sh lm.sh psd 10 3 0 0 1 0 0 0 0 $SEED
#sh lm.sh eds 10 3 0 0 1 0 0 0 0 $SEED
#sh lm.sh ptg 10 3 0 0 1 0 0 0 0 $SEED
#sh lm.sh ud 10 3 0 0 1 0 0 0 0 $SEED
#sh lm.sh ptb-phrase 10 3 0 0 1 0 0 0 0 $SEED                                  
#sh lm.sh ptb-pos 10 3 0 0 1 0 0 0 0 $SEED
#sh lm.sh ptb-func 10 3 0 0 1 0 0 0 0 $SEED
#sh lm.sh ptb-all 10 3 0 0 1 0 0 0 0 $SEED

#sh lm.sh dm 10 3 0 1 1 0 0 0 0 $SEED
#sh lm.sh psd 10 3 0 1 1 0 0 0 0 $SEED
#sh lm.sh eds 10 3 0 1 1 0 0 0 0 $SEED
#sh lm.sh ptg 10 3 0 1 1 0 0 0 0 $SEED
#sh lm.sh ud 10 3 0 1 1 0 0 0 0 $SEED
#sh lm.sh ptb-phrase 10 3 0 1 1 0 0 0 0 $SEED                                  
#sh lm.sh ptb-pos 10 3 0 1 1 0 0 0 0 $SEED
#sh lm.sh ptb-func 10 3 0 1 1 0 0 0 0 $SEED
#sh lm.sh ptb-all 10 3 0 1 1 0 0 0 0 $SEED


# LM only

#sh lm.sh dm 10 1 0 0 0 1 0 0 0 $SEED
#sh lm.sh psd 10 1 0 0 0 1 0 0 0 $SEED
#sh lm.sh eds 10 1 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptg 10 1 0 0 0 1 0 0 0 $SEED
#sh lm.sh ud 10 1 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptb-phrase 10 1 0 0 0 1 0 0 0 $SEED

#sh lm.sh dm 10 5 0 0 0 1 0 0 0 $SEED
#sh lm.sh psd 10 5 0 0 0 1 0 0 0 $SEED
#sh lm.sh eds 10 5 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptg 10 5 0 0 0 1 0 0 0 $SEED
#sh lm.sh ud 10 5 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptb-phrase 10 5 0 0 0 1 0 0 0 $SEED

# Graph only

#sh lm.sh dm 10 0 0 0 0 1 0 0 0 $SEED
#sh lm.sh psd 10 0 0 0 0 1 0 0 0 $SEED
#sh lm.sh eds 10 0 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptg 10 0 0 0 0 1 0 0 0 $SEED
#sh lm.sh ud 10 0 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptb-phrase 10 0 0 0 0 1 0 0 0 $SEED                                  
#sh lm.sh ptb-pos 10 0 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptb-func 10 0 0 0 0 1 0 0 0 $SEED
#sh lm.sh ptb-all 10 0 0 0 0 1 0 0 0 $SEED

# Graph only, no tokens

#sh lm.sh dm 10 0 0 0 0 1 0 0 1 $SEED
#sh lm.sh psd 10 0 0 0 0 1 0 0 1 $SEED
#sh lm.sh eds 10 0 0 0 0 1 0 0 1 $SEED
#sh lm.sh ptg 10 0 0 0 0 1 0 0 1 $SEED
#sh lm.sh ud 10 0 0 0 0 1 0 0 1 $SEED
#sh lm.sh ptb-phrase 10 0 0 0 0 1 0 0 1 $SEED                                  
#sh lm.sh ptb-pos 10 0 0 0 0 1 0 0 1 $SEED
#sh lm.sh ptb-func 10 0 0 0 0 1 0 0 1 $SEED
#sh lm.sh ptb-all 10 0 0 0 0 1 0 0 1 $SEED


