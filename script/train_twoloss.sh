python train_twoloss.py \
--pretrained_path='/home/ziyi/Desktop' \
--dataset_dir='/home/ziyi/share/Market-1501-v15.09.15/bounding_box_train/' \
--model_save_dir='/home/ziyi/Desktop/twoloss22/' \
--batch_size=32 \
--img_h=384 \
--img_w=128 \
--img_bi_h=384 \
--img_bi_w=128 \
--img_tri_h=576 \
--img_tri_w=192 \
--l2=0.002 \
--num_epochs=80
