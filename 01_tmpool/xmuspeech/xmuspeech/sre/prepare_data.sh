. ./cmd.sh
. ./path.sh
set -e

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
languages=local/general_lr_closed_set_langs.txt
data_root=data




local/make_lre03.pl $data_root/NIST_LR2003 data
local/make_lre05.pl $data_root/NIST_LR2005 data
local/make_lre07_train.pl $data_root/NIST_LR2007/train data
local/make_lre07.pl $data_root/NIST_LR2007/test/data/test data/lre07_all


cp -r data/lre07_all data/lre07
utils/filter_scp.pl -f 2 $languages <(lid/remove_dialect.pl data/lre07_all/utt2lang) \
  > data/lre07/utt2lang
utils/fix_data_dir.sh data/lre07

src_list="data/lid05d1 \
    data/lid05e1 data/lid96d1 data/lid96e1 data/lre03 \
    data/train_arabic data/train_bengali data/train_chinese.cantonese \
    data/train_chinese.min data/train_chinese.wu data/train_russian \
    data/train_thai data/train_urdu"


# Remove any spk2gender files that we have: since not all data
# sources have this info, it will cause problems with combine_data.sh
for d in $src_list; do rm -f $d/spk2gender 2>/dev/null; done

utils/combine_data.sh data/train_unsplit $src_list

# original utt2lang will remain in data/train_unsplit/.backup/utt2lang.
utils/apply_map.pl -f 2 --permissive local/lang_map.txt \
  < data/train_unsplit/utt2lang 2>/dev/null > foo
cp foo data/train_unsplit/utt2lang
rm foo

cp -r data/train_unsplit data/train
utils/filter_scp.pl -f 2 $languages <(lid/remove_dialect.pl data/train_unsplit/utt2lang) \
  > data/train/utt2lang
utils/fix_data_dir.sh data/train


# split lre07 to three pieces eg: lre07_test, lre07_dev, lre07_enroll
utils/split_data.sh $data_root/lre07 3
cd $data_root/lre07/split3
mv 1 ../../lre07_test
mv 2 ../../lre07_dev
mv 3 ../../lre07_enroll
cd ../
rm -r split3
cd ../../
#...


# add language prefix to utt2spk, spk2utt, utt2lang and wav.scp
# then delete "-c 1/2" from wav.scp
dirs="data/lre07_test data/lre07_dev data/lre07_enroll data/train"
for dir in $dirs; do
  rm $dir/utt2spk $dir/spk2utt
  cp $dir/utt2lang $dir/utt2lang_temp
  cp $dir/wav.scp $dir/wav_temp.scp
  rm $dir/wav.scp
  awk '{print $2}' $dir/utt2lang > $dir/utt2lang_temp_col2
  paste $dir/utt2lang_temp_col2 $dir/utt2lang_temp -d "_" > $dir/utt2spk
  paste $dir/utt2lang_temp_col2 $dir/wav_temp.scp -d "_" > $dir/wav.scp
  rm $dir/utt2lang
  cp $dir/utt2spk $dir/utt2lang
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
  rm $dir/utt2lang_temp $dir/utt2lang_temp_col2 $dir/wav_temp.scp


  # delete -c 1/2
  sed 's/-c 1 //g' $dir/wav.scp > $dir/wav_temp.scp
  sed 's/-c 2 //g' $dir/wav_temp.scp > $dir/wav.scp
  rm $dir/wav_temp.scp

  #fix dirs, make data ordered.
  utils/fix_data_dir.sh $dir
done

# # split lre07 to three pieces eg: lre07_test, lre07_dev, lre07_enroll
# utils/split_data.sh $data_root/lre07 3
# cd $data_root/lre07/split3
# mv 1 ../../lre07_test
# mv 2 ../../lre07_dev
# mv 3 ../../lre07_enroll
# cd ../
# rm -r split3
# cd ../../
# #...

echo "all done."