docker build -t xview2:37_pytorch14 -f Dockerfile-pytorch14-37 .
docker tag xview2:37_pytorch14 ekhvedchenya/xview2:37_pytorch14

docker run --rm --memory=7g --memory-swap=7g --memory-swappiness=0 --kernel-memory=7g --cpus=1^
    -v j:\xview2\test\images:/input^
    -v j:\xview2\test_predictions:/output^
    ekhvedchenya/xview2:37_pytorch14^
    /input/test_pre_00000.png /input/test_post_00000.png /output/test_localization_00000_pytorch14_v37.png /output/test_damage_00000_pytorch14_v37.png --color-mask --raw

docker run --rm --memory=7g --memory-swap=7g --memory-swappiness=0 --kernel-memory=7g --cpus=1^
    -v j:\xview2\test\images:/input^
    -v j:\xview2\test_predictions:/output^
    ekhvedchenya/xview2:37_pytorch14^
    /input/test_pre_00284.png /input/test_post_00284.png /output/test_localization_00284_pytorch14_v37.png /output/test_damage_00284_pytorch14_v37.png --color-mask --raw

docker run --rm --memory=7g --memory-swap=7g --memory-swappiness=0 --kernel-memory=7g --cpus=1^
    -v j:\xview2\test\images:/input^
    -v j:\xview2\test_predictions:/output^
    ekhvedchenya/xview2:37_pytorch14^
    /input/test_pre_00033.png /input/test_post_00033.png /output/test_localization_00033_pytorch14_v37.png /output/test_damage_00033_pytorch14_v37.png --color-mask --raw

docker run --rm --memory=7g --memory-swap=7g --memory-swappiness=0 --kernel-memory=7g --cpus=1^
    -v j:\xview2\test\images:/input^
    -v j:\xview2\test_predictions:/output^
    ekhvedchenya/xview2:37_pytorch14^
    /input/test_pre_00096.png /input/test_post_00096.png /output/test_localization_00096_pytorch14_v37.png /output/test_damage_00096_pytorch14_v37.png --color-mask --raw
