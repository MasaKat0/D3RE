mkdir -p ../log/D3RE_PU/ 2>/dev/null

EXPNAME='fmnist_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<10 ; i++))
	do
		folder_name=../log/D3RE_PU/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}
		python3 main_D3RE_PU.py fmnist fmnist_LeNet ${folder_name} ../data --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --normal_class ${INLIERCLASS} --seed $((2020+i)) --pi 0.333;
	done
done

EXPNAME='cifar10_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<10 ; i++))
	do
		folder_name=../log/D3RE_PU/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}
		python3 main_D3RE_PU.py cifar10 cifar10_LeNet ${folder_name} ../data --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --normal_class ${INLIERCLASS} --seed $((2020+i)) --pi 0.333;
	done
done

EXPNAME='mnist_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<10 ; i++))
	do
		folder_name=../log/D3RE_PU/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}
		python3 main_D3RE_PU.py mnist mnist_LeNet ${folder_name} ../data --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --normal_class ${INLIERCLASS} --seed $((2020+i)) --pi 0.333;
	done
done

