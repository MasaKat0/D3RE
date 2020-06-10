mkdir -p ../log/GT/ 2>/dev/null

#EXPNAME='fmnist_test'
#
#for ((class=0 ; class<10 ; class++))
#do
#	INLIERCLASS=${class}
#	for ((i=0 ; i<10 ; i++))
#	do
#		folder_name=../log/GT/${EXPNAME}_class${INLIERCLASS}_${i}th
#		mkdir -p ${folder_name} 2>/dev/null
#		echo ${folder_name}
#		python3 baseline_GT.py fmnist wrn ${folder_name} ../data --lr 0.001 --n_epochs 3 --lr_milestone 50 --batch_size 128 --weight_decay 0 --normal_class ${INLIERCLASS} --seed $((2020+i));
#	done
#done

EXPNAME='mnist_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<10 ; i++))
	do
		folder_name=../log/GT/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}
		python3 baseline_GT.py mnist wrn ${folder_name} ../data --lr 0.001 --n_epochs 3 --lr_milestone 50 --batch_size 128 --weight_decay 0 --normal_class ${INLIERCLASS} --seed $((2020+i));
	done
done

