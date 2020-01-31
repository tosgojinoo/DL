- [[Yolo] Darknet with pytorch](#-yolo--darknet-with-pytorch)
  * [PyTorch 구현 Github1](#pytorch-구현-github1)
    + [train option](#train-option)
    + [(참고) google colab 연동 시](#-참고--google-colab-연동-시-)
  * [PyTorch 구현 Github2](#pytorch-구현-github2)
  * [PyTorch 구현 Github3](#pytorch-구현-github3)
  

## PyTorch 구현 Github1
+ <https://github.com/eriklindernoren/PyTorch-YOLOv3>
```python
# YoloV3_Pytorch version git clone
!git clone https://github.com/eriklindernoren/PyTorch-YOLOv3

cd PyTorch-YOLOv3/

# 필요 패키지 설치
!pip3 install -r requirements.txt

# pillow 패키지 에러 발생시, version 확인과 재설치
!pip3 uninstall Pillow -y
!pip3 install Pillow

# pretrained weights 다운
cd PyTorch-YOLOv3/weights/
# 다운로드용 스크립트
cat /content/PyTorch-YOLOv3/weights/download_weights.sh
#다운
!bash download_weights.sh

# COCO 데이터셋 다운
$ cd PyTorch-YOLOv3/data/
$ bash get_coco_dataset.sh


# test
# 상위 폴더 이동
cd ..
#  --image_folder 옵션 -> 이미지들이 들어있는 폴더를 지정
!python3 detect.py — image_folder data/samples

# predict 대상 이미지 : /PyTorch-YOLOv3/data/samples/
# predict 결과 이미지 : /PyTorch-YOLOv3/output/ 
# 이미지 확인시
from IPython.display import Image
Image('/content/PyTorch-YOLOv3/data/samples/dog.jpg')

# 예측 이미지 확인
# ls -al [디렉토리/파일] : 파일 및 디렉토리의 전체 내용을 보여주는 명령어
ls -al /content/PyTorch-YOLOv3/output/
Image('/content/PyTorch-YOLOv3/output/3.png')


# train
# Example (COCO)
!python3 train.py --data_config config/coco.data  --pretrained_weights weights/darknet53.conv.74



```

### train option
```bash
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```

### (참고) google colab 연동 시
```python
# 라이브러리 설치 및 권한 부여를 위해 아래를 실행
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
# 디렉토리 생성 및 mount
!mkdir -p drive 
!google-drive-ocamlfuse drive
```
<br><br>


## PyTorch 구현 Github2
+ <https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch>


## PyTorch 구현 Github3
+ <https://github.com/dojinkimm/Object_Detection_Video_DNN_Tensorflow_Pytorch>








