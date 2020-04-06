python train_augnet.py \
--pretrained_path='/home/ziyi/Desktop' \
--pretrained_path1='/home/ziyi/Desktop/MSMTmodel' \
--pretrained_path2='/home/ziyi/Desktop/MSMTmodellarge' \
--pretrained_epoch1='net_119.pth' \
--pretrained_epoch2='net_139.pth' \
--dataset_dir='/home/ziyi/share/Market-1501-v15.09.15/bounding_box_train/' \
--model_save_dir='/home/ziyi/Desktop/augnettrypos/' \
--batch_size=16 \
--img_h=384 \
--img_w=128 \
--img_bi_h=576 \
--img_bi_w=192 \
--l2loss=0.001 \
--change_epoch=5 \
--num_epochs=100

