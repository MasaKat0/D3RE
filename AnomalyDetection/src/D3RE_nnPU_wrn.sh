mkdir -p ../log/D3RE_nnPU_wrn/ 2>/dev/null

EXPNAME='fmnist_test'

for ((class=1 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<5 ; i++))
	do
		folder_name=../log/D3RE_nnPU_wrn/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}
		python3 main_D3RE_nnPU.py fmnist wrn1 ${folder_name} ../data --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --normal_class ${INLIERCLASS} --seed $((2020+i)) --pi 0.333;
	done
done

#EXPNAME='cifar10_test'
#
#for ((class=0 ; class<10 ; class++))
#do
#	INLIERCLASS=${class}
#	for ((i=0 ; i<5 ; i++))
#	do
#		folder_name=../log/D3RE_nnPU_wrn/${EXPNAME}_class${INLIERCLASS}_${i}th
#		mkdir -p ${folder_name} 2>/dev/null
#		echo ${folder_name}
#		python3 main_D3RE_nnPU.py cifar10 wrn3 ${folder_name} ../data --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --normal_class ${INLIERCLASS} --seed $((2020+i)) --pi 0.333;
#	done
#done

EXPNAME='mnist_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<5 ; i++))
	do
		folder_name=../log/D3RE_nnPU_wrn/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}
		python3 main_D3RE_nnPU.py mnist wrn1 ${folder_name} ../data --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --normal_class ${INLIERCLASS} --seed $((2020+i)) --pi 0.333;
	done
done

