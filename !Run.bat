cd ZSSGAN
python train.py --size 1024 --batch 6 --n_sample 2 --output_dir out --lr 0.002  --frozen_gen_ckpt pretrained/stylegan2-ffhq-config-f.pt  --iter 201  --auto_layer_k 18 --auto_layer_iters 1  --auto_layer_batch 8  --output_interval 50  --clip_models "ViT-B/32" "ViT-B/16" --clip_model_weights 1.0 1.0 --mixing 0.0 --save_interval 50 --source_class "photo" --target_class "sketch"  

