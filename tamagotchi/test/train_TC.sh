#!/bin/bash

SHAPE="step oob"


BIRTHX="0.001" #  The lowest birthx to be reached
DATASET="constantx5b5 switch45x5b5 noisy3x5b5" # three DSs
STEPS="10000000" 
birthx_linear_tc_steps=7
VARX="2.0 1.0 0.5" # Variance of init. location; higher = more off-plume initializations. Note this is set to 0 in evalCli 
DMAX="0.8 0.8 0.8"
DMIN="0.7 0.65 0.4" 

ALGO=ppo
HIDDEN=64
DECAY=0.0001



RNNTYPE='VRNN'
eval_type='skip'
start_time=$(date +%s)

MAXJOBS=4
NUMPROC=8
# uses ~50% of GPU and 12.5% of GPU memory at n = 7
for SEED in $(seq 4); do  
  
    # EXPT=birthx_nsteps${birthx_linear_tc_steps}
    SAVEDIR=/src/tamagotchi/test/out
    mkdir -p $SAVEDIR
    while (( $(jobs -p | wc -l) >= MAXJOBS )); do sleep 10; done
    SEED=$RANDOM

    DATASTRING=$(echo -e $DATASET | tr -d ' ')
    SHAPESTRING=$(echo -e $SHAPE | tr -d ' ')
    BXSTRING=$(echo -e $BIRTHX | tr -d ' ')
    TSTRING=$(echo -e $STEPS | tr -d ' ')
    QVARSTR=$(echo -e $VARX | tr -d ' ')
    DMAXSTR=$(echo -e $DMAX | tr -d ' ')
    DMINSTR=$(echo -e $DMIN | tr -d ' ')

    OUTSUFFIX=$(date '+%Y%m%d')_${RNNTYPE}_${DATASTRING}_${SHAPESTRING}_bx${BXSTRING}_t${TSTRING}_q${QVARSTR}_dmx${DMAXSTR}_dmn${DMINSTR}_h${HIDDEN}_wd${DECAY}_n${NUMPROC}_code${RNNTYPE}_seed${SEED}$(openssl rand -hex 1)
    echo $OUTSUFFIX
    nice python3 -u /src/tamagotchi/main.py --env-name plume \
      --recurrent-policy \
      --dataset $DATASET \
      --num-env-steps ${STEPS} \
      --birthx $BIRTHX  \
      --flipping True \
      --qvar $VARX \
      --save-dir $SAVEDIR \
      --log-interval 1 \
      --r_shaping $(echo -e $SHAPE) \
      --algo $ALGO \
      --seed ${SEED} \
      --squash_action True \
      --diff_max $DMAX \
      --diff_min $DMIN \
      --num-processes $NUMPROC \
      --num-mini-batch $NUMPROC \
      --odor_scaling True \
      --rnn_type ${RNNTYPE} \
      --hidden_size $HIDDEN \
      --weight_decay ${DECAY} \
      --use-gae --num-steps 2048 --lr 3e-4 --entropy-coef 0.005 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay \
      --outsuffix ${OUTSUFFIX} --eval_type ${eval_type} --birthx_linear_tc_steps ${birthx_linear_tc_steps} > ${SAVEDIR}/${OUTSUFFIX}.log 2>&1 &

      echo "Sleeping.."
      sleep 0.5


done


# echo "some_command took $start_time seconds to run"
# echo "some_command took $end_time seconds to run"
# echo "some_command took $duration seconds to run"
# echo "some_command took $duration_hrs seconds to run"

# tail -f ${SAVEDIR}/*.log

wait 

end_time=$(date +%s) 
duration=$((end_time - start_time))
duration_hrs=$((duration/3600))

# check for any error in evallog - known error OSError: [Errno 107] Transport endpoint is not connected. 
for f in `find -name "*.evallog"`; do echo $f; less $f | grep "Error"; done
# check if all files here 
for FNAME in $FNAMES; do
    FOLDER=$(echo $FNAME | sed s/.pt//g)
    echo $FOLDER
    find $FOLDER -name "*csv" | wc -l
done