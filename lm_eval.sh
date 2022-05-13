FORML=$1
TAG=$2
MODL=$3

if [ -z "$4" ]
then
    UPOS=""
else
    UPOS="--eval-upos-file $4"
fi

if [ -z "$5" ]
then
    BASELINE=""
else
    BASELINE="--baseline-enc $5"
fi

if [ -z "$6" ]
then
    INPUT_UPOS=""
else
    INPUT_UPOS="--train-upos-file $6"
fi

CMD="python3 eval.py $FORML $TAG $MODL $UPOS $BASELINE $INPUT_UPOS"

JOB_NAME="lm-eval-$FORML$TAG$MODL-$5"
CMMND="eval-$FORML$TAG$MODL$5.cmd"
OUTPUT="eval-$FORML$TAG$MODL$5.out"
ERROR="eval-$FORML$TAG$MODL$5.err"
GRES="--gres=gpu:1"
NTASKS="1"
CPUS="8"
TIME="500:00"
MEM="24G"

read -r SBATCH_CMD << EOL
sbatch \
--job-name=$JOB_NAME \
--output=$OUTPUT \
--error=$ERROR \
$GRES \
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
