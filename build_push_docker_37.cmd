set mydate=%date:~10,4%%date:~4,2%%date:~7,2%

docker build -t xview2:37_pytorch14 -f Dockerfile-pytorch14-37 .
docker tag xview2:37_pytorch14 ekhvedchenya/xview2:37_pytorch14_%mydate%
START docker push ekhvedchenya/xview2:37_pytorch14_%mydate%