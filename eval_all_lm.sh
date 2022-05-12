#SEED=$1
SEED="14"

# Combined
sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED combined
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED combined mrp/upos.validation2.json
#sh lm_eval.sh ptb-all,ptb-all -10-0001-0.0_0.0-0-$SEED combined mrp/upos.validation2.json
#sh lm_eval.sh empty,empty,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED combined mrp/upos.validation2.json upos mrp/upos.training.json


# Combined PR
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-1.0_1.0-0-$SEED combined mrp/upos.validation2.json
#sh lm_eval.sh ptb-all,ptb-all -10-0001-0.0_1.0-0-$SEED combined mrp/upos.validation2.json

# Vanilla and finetuned LM
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase -0-0-$SEED,-10-0-$SEED lm mrp/upos.validation2.json

# Graph only
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0-$SEED graph mrp/upos.validation2.json

# Graph and combined, random permutation ablations
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED,-10-0101-0.0_0.0-0-$SEED combined
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED,-10-0011-0.0_0.0-0-$SEED combined
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED,-10-0010-0.0_0.0-0-$SEED combined
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED,-10-0110-0.0_0.0-0-$SEED combined
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED,-10-0101-0.0_0.0-0-$SEED graph
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED,-10-0011-0.0_0.0-0-$SEED graph
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED,-10-0010-0.0_0.0-0-$SEED graph
#sh lm_eval.sh dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func -10-0001-0.0_0.0-0-$SEED,-10-0110-0.0_0.0-0-$SEED graph

# Graph and combined, no graph tokens ablation
#sh lm_eval.sh dm,dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func,ptb-all -10-0001-1-$SEED combined
#sh lm_eval.sh dm,dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func,ptb-all -10-0001-1-$SEED graph
