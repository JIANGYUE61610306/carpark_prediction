
## Mounting the aws s3

Following the instruction of [s3fs](https://github.com/s3fs-fuse/s3fs-fuse) to setup and it can be configured with the `s3fs_config.sh`

```
bash s3fs_config.sh
```

Run the following command to mount the aws s3 to the local file system,

```
mkdir ~/s3drive
s3fs parkingdata ~/s3drive -o umask=0007,uid=xxx
```

where the `uid` should be your user id and can be obtained by `id` command.

## Create container 
using docker run -it --name [your container name] --shm-size 16G --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 [image name]

## Switch envs
conda activate pytorch-gpu

## Preprocessing

Preprocessing the raw json data,

```
python transform_carpark.py
```

or

```
python transform_carpark.py --month 5 --startday 1 --endday 10
```

## Training

Training the model by using the `carpark.ipynb` to produce the `model.pt`.
For container envs, using python 'carpark_pre.py' to produce the 'model.pt'

## More on s3fs

umount the aws s3

```
fusermount -u ~/s3drive
```

