FORML=$1
TAG=$2
MODL=$3
UPOS=$4
BASELINE=$5
INPUT_UPOS=$6

CMD="python3 eval.py $FORML $TAG $MODL $UPOS --baseline-enc $BASELINE --train-upos-file $INPUT_UPOS"

FORML=$(echo $FORML | sed -r 's/(,?)uos\/([^,]*)/\1\2/g')

JOB_NAME="lm-eval-$FORML$TAG$MODL-$BASELINE"
CMMND="eval-$FORML$TAG$MODL$BASELINE.cmd"
OUTPUT="eval-$FORML$TAG$MODL$BASELINE.out"
ERROR="eval-$FORML$TAG$MODL$BASELINE.err"
GRES="--gres=gpu:1"
NTASKS="1"
CPUS="8"
TIME="500:00"
MEM="24G"

read -r -d '' SBATCH_CMD << EOM
sbatch --job-name=$JOB_NAME
       --output=$OUTPUT
       --error=$ERROR
       $GRES
       --ntasks=$NTASKS
       --cpus-per-task=$CPUS
       --time=$TIME
       --mem=$MEM
       --wrap="$CMD"
EOM

echo "$SBATCH_CMD" | tee "$CMMND"

eval $SBATCH_CMD

echo "Running on" `hostname`
