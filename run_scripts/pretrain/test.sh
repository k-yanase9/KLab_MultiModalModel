batch_size=128

epoch=20

enc=2
dec=12

dataset="sun397"
python test.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase pretrain \
        --loss CrossEntropy \
        -b $batch_size \
        --num_epochs $epoch \
        --datasets $dataset \
        --root_dir /local/ \
        --result_dir results/pretrain/cc3m_cc12m_imagenet_imagenet21k_places365_redcaps_sun397/new/enc$enc\_dec$dec/Linear$epoch/

for dataset in "imagenet21k" "cc12m"; do
python test.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase pretrain \
        --loss CrossEntropy \
        -b $batch_size \
        --num_epochs $epoch \
        --datasets $dataset \
        --root_dir /data01/ \
        --result_dir results/pretrain/cc3m_cc12m_imagenet_imagenet21k_places365_redcaps_sun397/new/enc$enc\_dec$dec/Linear$epoch/
done