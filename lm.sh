FORML=$1
EPOCHS=$2
MODE=$3

SCRATCH=$4
MIX_LAB=$5
MIX_ANC=$6
KEEP_UNA=$7

LM_WEIGHT=$8
AUX_WEIGHT=$9

NOTOK=${10}

SEED=${11}


if [ -z "${12}" ]
then
    BASELINE=""
else
    BASELINE="--baseline-enc ${12}"
fi

if [ -z "${13}" ]
then
    UPOS=""
else
    UPOS="--upos-file ${13}"
fi

CMD="python3 train.py $FORML $EPOCHS $MODE $SCRATCH $MIX_LAB $MIX_ANC $KEEP_UNA $LM_WEIGHT $AUX_WEIGHT $NOTOK $SEED --baseline-enc $BASELINE --upos-file $UPOS"

JOB_NAME="lm-$FORML-$MODE-$SCRATCH$MIX_LAB$MIX_ANC$KEEP_UNA-$LM_WEIGHT$AUX_WEIGHT-$NOTOK-$SEED${12}"
CMMND="$FORML-$MODE-$SCRATCH$MIX_LAB$MIX_ANC$KEEP_UNA-$LM_WEIGHT$AUX_WEIGHT-$NOTOK.cmd"
OUTPUT="$FORML-$MODE-$SCRATCH$MIX_LAB$MIX_ANC$KEEP_UNA-$LM_WEIGHT$AUX_WEIGHT-$NOTOK-$SEED${12}.out"
ERROR="$FORML-$MODE-$SCRATCH$MIX_LAB$MIX_ANC$KEEP_UNA-$LM_WEIGHT$AUX_WEIGHT-$NOTOK-$SEED${12}.err"
GRES="gpu:1"
NTASKS="1"
CPUS="8"
TIME="3-00:00"
MEM="28G"

read -r $BATCH_CMD << EOL
sbatch --job-name=$JOB_NAME \
       --output=$OUTPUT \
       --error=$ERROR \
       --gres=$GRES \
       --ntasks=$NTASKS \
       --cpus-per-task=$CPUS \
       --time=$TIME \
       --mem=$MEM \
       --wrap="$CMD"
EOL


echo "$SBATCH_CMD" | tee "$CMMND"

{
    eval $SBATCH_CMD &&
    echo "Running on" `hostname`
} || {
    echo "slurm not found; running in shell instead..."
    eval $CMD > $OUTPUT 2> $ERROR
}
