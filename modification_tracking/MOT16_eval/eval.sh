#!/bin/bash

set +e


# start from clean slate
for i in data.zip MOT16.zip
do
	zip -T $i
	if [ $? -eq 0 ]
	then
		echo 'zip is ok'

	else
		echo 'zip corrupted, deleting'
		rm -rf $i
	fi
done


# create output folder if it doesn't exist
if [ ! -d ~/yolov8_tracking/inference/output ]
then
	mkdir -p ~/yolov8_tracking/inference/output
	echo 'inference output folder created'
fi



# clone evaluation repo if it does not exist
if [ ! -d ~/yolov8_tracking/MOT16_eval/TrackEval ]
then
	echo 'Cloning official MOT16 evaluation repo'
	git clone https://github.com/JonathonLuiten/TrackEval ~/yolov8_tracking/MOT16_eval/TrackEval
	# download quick start data folder if it does not exist
	if [ ! -d ~/yolov8_tracking/MOT16_eval/TrackEval/data ]
	then
		# download data
		wget -nc https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip -O ~/yolov8_tracking/data.zip
		# unzip
		unzip -q ~/yolov8_tracking/data.zip -d ~/yolov8_tracking/MOT16_eval/TrackEval/
		# delete zip
		#rm data.zip
	fi
fi


# create MOT16 folder if it doesn't exist
if [ ! -d ~/yolov8_tracking/MOT16_eval/TrackEval/data/MOT16 ]
then
	mkdir -p ~/yolov8_tracking/MOT16_eval/TrackEval/data/MOT16
fi


# if MOT16 data not unziped, then download, unzip and lastly remove zip MOT16 data
if [[ ! -d ~/yolov8_tracking/MOT16_eval/TrackEval/data/MOT16/train ]] && [[ ! -d ~/Yolov5_DeepSort_Pytorch/MOT16_eval/TrackEval/data/MOT16/test ]]
then
	# download data
	wget -nc https://motchallenge.net/data/MOT16.zip -O ~/yolov8_tracking/MOT16.zip
	# unzip
    unzip -q ~/yolov8_tracking/MOT16.zip -d ~/yolov8_tracking/MOT16_eval/TrackEval/data/MOT16/
	# delete zip
	#rm MOT16.zip
fi


# create folder to place tracking results for this method
mkdir -p ~/yolov8_tracking/MOT16_eval/TrackEval/data/trackers/mot_challenge/MOT16-train/ch_yolov5m_sota_tracking/data/

# inference on 4 MOT16 video sequences at the same time
# suits a 4GB GRAM GPU, feel free to increase if you have more memory
N=1

# generate tracking results for each sequence
for i in MOT16-02 MOT16-04 MOT16-05 MOT16-09 MOT16-10 MOT16-11 MOT16-13
do
	(
		# change name to inference source so that each thread write to its own .txt file
		if [ ! -d ~/yolov8_tracking/MOT16_eval/TrackEval/data/MOT16/train/$i/$i ]
		then
			mv ~/yolov8_tracking/MOT16_eval/TrackEval/data/MOT16/train/$i/img1/ ~/yolov8_tracking/MOT16_eval/TrackEval/data/MOT16/train/$i/$i
		fi
		# run inference on sequence frames
		python3 track_yolov5.py --tracking-method strongsort --source ~/yolov8_tracking/MOT16_eval/TrackEval/data/MOT16/train/$i/$i --save-txt --yolo-weights yolov5/weights/crowdhuman_yolov5m.pt --classes 0 --exist-ok
	    # move generated results to evaluation repo
	) &
	# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
	# allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]
	then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi
done

# no more jobs to be started but wait for pending jobs
# (all need to be finished)
wait
echo "Inference on all MOT16 sequences DONE"

echo "Moving data from experiment folder to MOT16"
mv ~/yolov8_tracking/runs/track/exp/* \
   ~/yolov8_tracking/MOT16_eval/TrackEval/data/trackers/mot_challenge/MOT16-train/ch_yolov5m_sota_tracking/data/

# run the evaluation
python ~/yolov8_tracking/MOT16_eval/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 \
 --TRACKERS_TO_EVAL ch_yolov5m_sota_tracking --SPLIT_TO_EVAL train --METRICS CLEAR Identity \
 --USE_PARALLEL False --NUM_PARALLEL_CORES 4
