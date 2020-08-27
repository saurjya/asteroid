#!/bin/bash
set -e  # Exit on error

#if starting from stage 0
# Destination to save json files with list of track locations for instrument sets
json_dir=/homes/ss404/projects/asteroid/egs/MedleyDB/ConvTasNet/data/1inst2poly44sr5.0sec
# Location for tracklist for all data dirs
tracklist=  # Directory containing tracklists for V1, V2, Bach10 and others

# Location for MedleyDB V1
V1_dir="/import/research_c4dm/ss404/V1" # Directory containing MedleyDB V1 audio files

# Location for MedleyDB V2
V2_dir="/import/research_c4dm/ss404/V2" # Directory containing MedleyDB V2 audio files

# Location for Bach10
Bach10_dir=  # Directory containing MedleyDB format Bach10 audio files

# Location for additional MedleyDB format multitracks
extra_dir= # Directory containing additional MedleyDB format audio files

# Location for MedleyDB format metadata files for all multitracks
metadata_dir=/homes/ss404/projects/medleydb # Directory containing MedleyDB github repository with metadata for all files


# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --loss_alpha 0.1 --id 0,1

# General
stage=3  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Data
#data_dir=data  # Local data directory (No disk space needed)
sample_rate=44100
n_inst=1  # 2 or 3
n_poly=3
segment=5.0
# Training
batch_size=1
num_workers=10
optimizer=rmsprop
lr=0.0001
weight_decay=0.0
epochs=200
loss_alpha=1.0  # DC loss weight : 1.0 => DC, <1.0 => Chimera
take_log=true  # Whether to input log mag spec to the NN

# Evaluation
eval_use_gpu=1

. utils/parse_options.sh

sr_string=$(($sample_rate/1000))
suffix=${n_inst}inst${n_poly}poly${sr_string}sr${segment}sec
dumpdir=data/$suffix  # directory to put generated json file

#json_dir=$dumpdir
is_raw=True

if [[ $stage -le  0 ]]; then
  echo "Stage 0: Converting sphere files to wav files"
  $python_path local/preprocess_medleyDB.py --metadata_path $metadata_dir --json_dir $json_dir --v1_path "$V1_dir" --v2_path "$V2_dir"
fi

if [[ $stage -le  1 ]]; then
	echo "Stage 1 : Downloading wsj0-mix mixing scripts"
	# Link + WHAM is ok for 2 source.
	#wget https://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip -O ./local/
	#unzip ./local/create-speaker-mixtures.zip -d ./local/create-speaker-mixtures
	#mv ./local/create-speaker-mixtures.zip ./local/create-speaker-mixtures

	echo "You need to generate the wsj0-mix dataset using the official MATLAB
			  scripts (already downloaded into ./local/create-speaker-mixtures).
			  If you don't have Matlab, you can use Octavve and replace
				all mkdir(...) in create_wav_2speakers.m with system(['mkdir -p '...]).
				Note: for 2-speaker separation, the sep_clean task from WHAM is the same as
				wsj0-2mix and the mixing scripts are in Python.
				Specify wsj0mix_wav_dir and start from stage 2 when the mixtures have been generated.
				Exiting now."
	#exit 1
fi

if [[ $stage -le  2 ]]; then
	# Make json directories with min/max modes and sampling rates
	echo "Stage 2: Generating json files including wav path and duration"
	#for sr_string in 8 16; do
	#	for mode_option in min max; do
	#		for tmp_nsrc in 2 3; do
	#			tmp_dumpdir=data/${tmp_nsrc}speakers/wav${sr_string}k/$mode_option
	#			echo "Generating json files in $tmp_dumpdir"
	#			[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
	#			local_wsj_dir=$wsj0mix_wav_dir/${tmp_nsrc}speakers/wav${sr_string}k/$mode_option/
	#			$python_path local/preprocess_wsj0mix.py --in_dir $local_wsj_dir \
	#			 																			--n_src $tmp_nsrc \
	#			 																			--out_dir $tmp_dumpdir
	#		done
    #done
  #done
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${n_src}sep_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--json_dir $json_dir \
		--n_inst $n_inst \
		--sample_rate $sample_rate \
		--optimizer $optimizer \
		--lr $lr \
		--weight_decay $weight_decay \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	#CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
	#  --n_src $n_src \
	#	--test_dir $test_dir \
	#	--use_gpu $eval_use_gpu \
	#	--exp_dir ${expdir} | tee logs/eval_${tag}.log
	#cp logs/eval_${tag}.log $expdir/eval.log
fi
