mkdir -p ../log/SSAD/ 2>/dev/null

EXPNAME='fmnist_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<10 ; i++))
	do
		folder_name=../log/SSAD/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}

		python baseline_ssad.py fmnist ${folder_name} ../data --kernel rbf --kappa 1 --normal_class ${INLIERCLASS} --seed $((2020+i));
	done
done

EXPNAME='cifar10_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<10 ; i++))
	do
		folder_name=../log/SSAD/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}

		python baseline_ssad.py cifar10 ${folder_name} ../data --kernel rbf --kappa 1 --normal_class ${INLIERCLASS} --seed $((2020+i));
	done
done

EXPNAME='mnist_test'

for ((class=0 ; class<10 ; class++))
do
	INLIERCLASS=${class}
	for ((i=0 ; i<10 ; i++))
	do
		folder_name=../log/SSAD/${EXPNAME}_class${INLIERCLASS}_${i}th
		mkdir -p ${folder_name} 2>/dev/null
		echo ${folder_name}
		python baseline_ssad.py mnist ${folder_name} ../data --kernel rbf --kappa 1.0 --normal_class ${INLIERCLASS} --seed $((2020+i));
	done
done

