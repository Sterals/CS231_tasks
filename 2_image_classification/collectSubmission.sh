#!/bin/bash
#NOTE: DO NOT EDIT THIS FILE-- MAY RESULT IN INCOMPLETE SUBMISSIONS

NOTEBOOKS="knn.ipynb 
svm.ipynb 
softmax.ipynb 
two_layer_net.ipynb 
features.ipynb"

CODE="cs231n/classifiers/k_nearest_neighbor.py
cs231n/classifiers/linear_classifier.py
cs231n/classifiers/linear_svm.py
cs231n/classifiers/softmax.py
cs231n/classifiers/neural_net.py"

LOCAL_DIR=`pwd`
REMOTE_DIR="cs231n-2019-assignment1"
ASSIGNMENT_NO=1
ZIP_FILENAME="a1.zip"

C_R="\e[31m"
C_G="\e[32m"
C_BLD="\e[1m"
C_E="\e[0m"

FILES=""
for FILE in "${NOTEBOOKS} ${CODE}"
do
	if [ ! -f ${F} ]; then
		echo -e "${C_R}Required file ${FILE} not found, Exiting.${C_E}"
		exit 0
	fi
	FILES="${FILES} ${LOCAL_DIR}/${FILE}"
done

echo -e "${C_BLD}### Zipping file ###${C_E}"
rm -f ${ZIP_FILENAME}
zip -r ${ZIP_FILENAME} . -x "*.git*" "*cs231n/datasets*" "*.ipynb_checkpoints*" "*README.md" "collectSubmission.sh" "*requirements.txt" "*__pycache__*" ".env/*" > assignment_zip.log
echo ""

echo -e "${C_BLD}### Submitting to myth ###${C_E}"
echo "Type in your Stanford student ID (alphanumeric, *not* the 8-digit ID):"
read -p "Student ID: " SUID
echo ""

echo -e "${C_BLD}### Copying to ${SUID}@myth.stanford.edu:${REMOTE_DIR} ###${C_E}"
echo -e "${C_G}Note: if myth is under heavy use, this may hang: If this happens, rerun the script.${C_E}"
FILES="${FILES} ${LOCAL_DIR}/${ZIP_FILENAME}"
rsync -avP ${FILES} ${SUID}@myth.stanford.edu:${REMOTE_DIR}
echo ""

echo -e "${C_BLD}### Running remote submission script from ${SUID}@myth.stanford.edu:${REMOTE_DIR} ###${C_E}"
ssh ${SUID}@myth.stanford.edu "cd ${REMOTE_DIR} && /afs/ir/class/cs231n/grading/submit ${ASSIGNMENT_NO} ${SUID} ${ZIP_FILENAME} && exit"