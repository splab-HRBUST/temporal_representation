
#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2020-03-25)

type="xvector_score"

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# != 1 ]];then
echo "[exit] Num of parameters is not equal to 1"
echo "usage: $0 --type xvector_score <find-dir>"
exit 1
fi

dir=$1

if [ "$type" == "xvector_score" ];then
    # Clear the files generated by back-end scoring process. It will not clear the original xvector.ark or xvector.scp.
    find $dir -name "xvector_*" | xargs -n 1 rm -f
else
    echo "Do not support $type now." && exit 1
fi

echo "Clear done."