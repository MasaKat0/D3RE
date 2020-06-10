mkdir -p ../log/DeepSAD/ 2>/dev/null


EXPNAME='fmnist_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<10 ; i++))
	do
		folder_name=../log/DeepSAD/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}
		python3 baseline_sad.py fmnist fmnist_LeNet ${folder_name} ../data --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class ${INLIERCLASS} --seed $((2020+i));
	done
done

EXPNAME='cifar10_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<10 ; i++))
	do
		folder_name=../log/DeepSAD/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}
		python baseline_sad.py cifar10 cifar10_LeNet ${folder_name} ../data --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class ${INLIERCLASS} --seed $((2020+i));
	done
done

EXPNAME='mnist_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<10 ; i++))
	do
		folder_name=../log/DeepSAD/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}
		python baseline_sad.py mnist mnist_LeNet ${folder_name} ../data --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class ${INLIERCLASS} --seed $((2020+i));
	done
done

