# Step 0
python convert_masks.py -dd $XVIEW2_DATA_DIR

# Train base models (no pseudolabeling, no optimized weights)
# Batch size tuned to fit into p3.8xlarge instance or 4x1080Ti
# Estimated training time ~1 week

# Step 1
python fit_predict.py --seed 330  -dd $XVIEW2_DATA_DIR -x fold0_resnet34_unet_v2 --model resnet34_unet_v2         --batch-size 64 --epochs 150 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 0 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True
python fit_predict.py --seed 332  -dd $XVIEW2_DATA_DIR -x fold0_resnet101_fpncatv2_256 --model resnet101_fpncatv2_256   --batch-size 48 --epochs 150 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 0 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True

python fit_predict.py --seed 13   -dd $XVIEW2_DATA_DIR -x fold1_seresnext50_unet_v2 --model seresnext50_unet_v2      --batch-size 32 --epochs 150 --learning-rate 0.001  --criterion [['weighted_ce', '1'], ['focal', '1']]  -w 16 -a medium --fp16  --fold 1 --scheduler simple -wd 1e-05   --only-buildings True  --crops True --post-transform True
python fit_predict.py --seed 331  -dd $XVIEW2_DATA_DIR -x fold1_resnet34_unet_v2 --model resnet34_unet_v2         --batch-size 64 --epochs 150 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 1 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True
python fit_predict.py --seed 1331 -dd $XVIEW2_DATA_DIR -x fold1_densenet201_fpncatv2_256 --model densenet201_fpncatv2_256 --batch-size 32 --epochs 150 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 1 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True

python fit_predict.py --seed 333  -dd $XVIEW2_DATA_DIR -x fold2_inceptionv4_fpncatv2_256 --model inceptionv4_fpncatv2_256 --batch-size 48 --epochs 150 --learning-rate 0.001  --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 2 --scheduler poly   -wd 1e-05   --only-buildings True  --crops True --post-transform True
python fit_predict.py --seed 303  -dd $XVIEW2_DATA_DIR -x fold2_densenet169_unet_v2 --model densenet169_unet_v2      --batch-size 32 --epochs 150 --learning-rate 0.001  --criterion [['ohem_ce', '1']]                      -w 16 -a medium --fp16  --fold 2 --scheduler simple -wd 1e-05   --only-buildings True  --crops True --post-transform True
python fit_predict.py --seed 332  -dd $XVIEW2_DATA_DIR -x fold2_resnet34_unet_v2 --model resnet34_unet_v2         --batch-size 64 --epochs 150 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 2 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True

python fit_predict.py --seed 50   -dd $XVIEW2_DATA_DIR -x fold3_resnet34_unet_v2 --model resnet34_unet_v2         --batch-size 16 --epochs 150 --learning-rate 0.0003 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 3 --scheduler cos    -wd 1e-05   --only-buildings True  --crops True --post-transform True
python fit_predict.py --seed 3    -dd $XVIEW2_DATA_DIR -x fold3_seresnext50_unet_v2 --model seresnext50_unet_v2      --batch-size 32 --epochs 150 --learning-rate 0.001  --criterion [['weighted_ce', '1'], ['focal', '1']]  -w 16 -a medium --fp16  --fold 3 --scheduler simple -wd 1e-05   --only-buildings True  --crops True --post-transform True
python fit_predict.py --seed 3334 -dd $XVIEW2_DATA_DIR -x fold3_efficientb4_fpncatv2_256 --model efficientb4_fpncatv2_256 --batch-size 32 --epochs 150 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 3 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True

python fit_predict.py --seed 334  -dd $XVIEW2_DATA_DIR -x fold4_resnet34_unet_v2 --model resnet34_unet_v2         --batch-size 64 --epochs 150 --learning-rate 0.001  --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 4 --scheduler cos    -wd 1e-05   --only-buildings True  --crops True --post-transform True
python fit_predict.py --seed 133  -dd $XVIEW2_DATA_DIR -x fold4_resnet101_unet_v2 --model resnet101_unet_v2        --batch-size 40 --epochs 150 --learning-rate 0.001  --criterion [['weighted_ce', '1'], ['focal', '1']]  -w 16 -a medium --fp16  --fold 4 --scheduler simple -wd 0.0001  --only-buildings True  --crops True --post-transform True

# Run inference on test dataset
python predict.py -dd $XVIEW2_DATA_DIR -tta flipscale -p naive -o stage1_predictions -b 16 --fp16 \
     fold0_resnet34_unet_v2.pth\
     fold0_resnet101_fpncatv2_256.pth\
     fold1_seresnext50_unet_v2.pth\
     fold1_resnet34_unet_v2.pth\
     fold1_densenet201_fpncatv2_256.pth\
     fold2_inceptionv4_fpncatv2_256.pth\
     fold2_densenet169_unet_v2.pth\
     fold2_resnet34_unet_v2.pth\
     fold3_resnet34_unet_v2.pth\
     fold3_seresnext50_unet_v2.pth\
     fold3_efficientb4_fpncatv2_256.pth\
     fold4_resnet34_unet_v2.pth\
     fold4_resnet101_unet_v2.pth

# Step 2
# Fine-tune using pseudo-label predictions
# Estimated training time ~3-4 days
python finetune.py --seed 330  -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling  -x fold0_pl_resnet34_unet_v2 --model resnet34_unet_v2         --batch-size 64 --epochs 50 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 0 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True
python finetune.py --seed 332  -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold0_pl_resnet101_fpncatv2_256 --model resnet101_fpncatv2_256   --batch-size 48 --epochs 50 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 0 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True

python finetune.py --seed 13   -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold1_pl_seresnext50_unet_v2 --model seresnext50_unet_v2      --batch-size 32 --epochs 50 --learning-rate 0.0001  --criterion [['weighted_ce', '1'], ['focal', '1']]  -w 16 -a medium --fp16  --fold 1 --scheduler simple -wd 1e-05   --only-buildings True  --crops True --post-transform True
python finetune.py --seed 331  -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold1_pl_resnet34_unet_v2 --model resnet34_unet_v2         --batch-size 64 --epochs 50 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 1 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True
python finetune.py --seed 1331 -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold1_pl_densenet201_fpncatv2_256 --model densenet201_fpncatv2_256 --batch-size 32 --epochs 50 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 1 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True

python finetune.py --seed 333  -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold2_pl_inceptionv4_fpncatv2_256 --model inceptionv4_fpncatv2_256 --batch-size 48 --epochs 50 --learning-rate 0.001  --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 2 --scheduler poly   -wd 1e-05   --only-buildings True  --crops True --post-transform True
python finetune.py --seed 303  -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold2_pl_densenet169_unet_v2 --model densenet169_unet_v2      --batch-size 32 --epochs 50 --learning-rate 0.0001  --criterion [['ohem_ce', '1']]                      -w 16 -a medium --fp16  --fold 2 --scheduler simple -wd 1e-05   --only-buildings True  --crops True --post-transform True
python finetune.py --seed 332  -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold2_pl_resnet34_unet_v2 --model resnet34_unet_v2         --batch-size 64 --epochs 50 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 2 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True

python finetune.py --seed 50   -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold3_pl_resnet34_unet_v2 --model resnet34_unet_v2         --batch-size 16 --epochs 50 --learning-rate 0.0003 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 3 --scheduler cos    -wd 1e-05   --only-buildings True  --crops True --post-transform True
python finetune.py --seed 3    -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold3_pl_seresnext50_unet_v2 --model seresnext50_unet_v2      --batch-size 32 --epochs 50 --learning-rate 0.0001  --criterion [['weighted_ce', '1'], ['focal', '1']]  -w 16 -a medium --fp16  --fold 3 --scheduler simple -wd 1e-05   --only-buildings True  --crops True --post-transform True
python finetune.py --seed 3334 -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold3_pl_efficientb4_fpncatv2_256 --model efficientb4_fpncatv2_256 --batch-size 32 --epochs 50 --learning-rate 0.0001 --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 3 --scheduler cos    -wd 0.0001  --only-buildings True  --crops True --post-transform True

python finetune.py --seed 334  -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold4_pl_resnet34_unet_v2 --model resnet34_unet_v2         --batch-size 64 --epochs 50 --learning-rate 0.0001  --criterion [['weighted_ce', '1']]                  -w 16 -a medium --fp16  --fold 4 --scheduler cos    -wd 1e-05   --only-buildings True  --crops True --post-transform True
python finetune.py --seed 133  -dd $XVIEW2_DATA_DIR -pl stage1_predictions_pseudolabeling -x fold4_pl_resnet101_unet_v2 --model resnet101_unet_v2        --batch-size 40 --epochs 50 --learning-rate 0.0001  --criterion [['weighted_ce', '1'], ['focal', '1']]  -w 16 -a medium --fp16  --fold 4 --scheduler simple -wd 0.0001  --only-buildings True  --crops True --post-transform True


# Make OOF predictions on fine-tuned models
# This would require up to 1Tb to save raw masks in NPY format
python predict_off.py -dd $XVIEW2_DATA_DIR\
    fold0_pl_resnet34_unet_v2.pth\
    fold0_pl_resnet101_fpncatv2_256.pth\
    fold1_pl_seresnext50_unet_v2.pth\
    fold1_pl_resnet34_unet_v2.pth\
    fold1_pl_densenet201_fpncatv2_256.pth\
    fold2_pl_inceptionv4_fpncatv2_256.pth\
    fold2_pl_densenet169_unet_v2.pth\
    fold2_pl_resnet34_unet_v2.pth\
    fold3_pl_resnet34_unet_v2.pth\
    fold3_pl_seresnext50_unet_v2.pth\
    fold3_pl_efficientb4_fpncatv2_256.pth\
    fold4_pl_resnet34_unet_v2.pth\
    fold4_pl_resnet101_unet_v2.pth

# Optimize per-class weights. As a result, you will get optimized_weights_%timestamp%.csv file 
# This is very CPU and IO consuming operation
# Exhaustive search of optimal weights checkpoint may take up to several hours PER ONE checkpoint.
python optimize_softmax.py -dd $XVIEW2_DATA_DIR\
    fold0_pl_resnet34_unet_v2.pth\
    fold0_pl_resnet101_fpncatv2_256.pth\
    fold1_pl_seresnext50_unet_v2.pth\
    fold1_pl_resnet34_unet_v2.pth\
    fold1_pl_densenet201_fpncatv2_256.pth\
    fold2_pl_inceptionv4_fpncatv2_256.pth\
    fold2_pl_densenet169_unet_v2.pth\
    fold2_pl_resnet34_unet_v2.pth\
    fold3_pl_resnet34_unet_v2.pth\
    fold3_pl_seresnext50_unet_v2.pth\
    fold3_pl_efficientb4_fpncatv2_256.pth\
    fold4_pl_resnet34_unet_v2.pth\
    fold4_pl_resnet101_unet_v2.pth

