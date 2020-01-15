# [Yolo v3] Source Code

## General
- [소스 확인](https://github.com/pjreddie/darknet)
- C언어로 구성

## 폴더 구조
```sh
darknet
  |
  |- backup/
  |- cfg/
  |- data/
  |- examples/
  |- LICENSE
  |- Makefile
  |- obj/
  |- README.md
  |- results/
  |- scripts/ 
  |- src/
```
+ `cfg`(configure) : 모델 튜닝, 데이터 적용
  + coco.data, yolov3.cfg(v3), yolo-obj.cfg(v2), rnn.cfg 등
+ `data` : bounding box 그릴때 필요한 label들의 폰트, label list, test image
  + coco.names : 기본 데이터셋인 coco.data 의 class 확인 가능 (80개)
+ `src`(source) : c 코드, 헤더파일 등
  + image.c : bounding box의 좌표에 대한 코드 <br>


## Makefile
콘솔 응용프로그램에서는 main 부터 찾는 것이 편함
makefile 확인하여 main 포함된 소스코드 확인 가능
+ `GPU=1` : CUDA 설치 시 1, 미설치 0
+ `OPENCV=1` : CUDA 설치 시 1, 미설치 0

+ `VPATH=./src/:./examples` : 경로 설정, 소스파일 위치, examples
+ `EXEC=darknet` : main source file (int main) 명
+ `OBJDIR=./obj/` : build 시 생성되는 object 파일들은 해당 경로에 출력됨

+ `CC` : C Compiler
+ `NVCC=nvcc` : CUDA compiler
+ `LDFLAGS` : linker option
+ `CFLAGS` : compile option <br>


## darknet/src/Darknet.c (Main source)
+ `detect` : 단일/멀티 이미지용, `test_detector` 함수 실행(detector.c 내부에 있음)
  + *Option*
    + `-ext_output` : 결과에 대한 상세 정보 표시
    + `-i 1` : Use GPU 1
    + `thresh 0.25 -dont_show -save_labels < list.txt` : 
      + YOLO 임계치 조정
      + 원래 25% 이상인 물체만 표시
      + 해당 옵션을 이용하여 0% 이상인 모든 물체를 표시하게 하거나, 50% 이상의 물체만 탐지를 하는 등의 설정 가능
    + List of Image에 대한 결과 저장

+ `detector` : 동영상용 <br>



## detector.c
+ `draw_detections`: detection 결과 표현, 갯수 출력, 인식 물체 이름 등 정의  <br>



## yolo.c
+ `char *voc_names[]` : 클래스 이름 설정 변수
+ `char *train_images` : 학습할 image들의 list.txt파일 위치
+ `char *backup_directory` : 학습을 하면서 중간결과들을 저장해놓는 폴더 위치, 최종 가중치 파일 저장 위치 동일 <br>



## (obj).cfg
모델 구조 및 train과 관련된 설정 포함 <br>

+ data feed
  + `batch = 64`
  + `subdivision = 8`
    + Cuda memory 관련
    + 배치 사이즈를 얼마나 쪼개서 학습할 것인지에 대한 설정
    + Out of memory 오류 시, subdivision 을 16, 32, 64 등으로 증가시킴
  + `height/width = 416`(32의 배수, 608 or 832 추천, 클수록 정확도 향상)

+ augmentation
  + `angle`: 이미지 회전 정도 설정, 보통은 0, 경우에 따라 90도 가능
  + `saturation` : 채도 추가 원할 때 설정
  + `expose` : 노출 값 추가 원할 때 설정
  + `hue` : 색상 변경 원할 때 설정

+ training
  + `learning rate = 0.001` 
    + muti-gpu 사용 시 학습율 0.001/gpu 수만큼 조절하기도 함
  + `burn_in = 1000`
    + multi-gpu 사용 시 몇 만큼의 iteration 지날 때 마다 학습률을 조장 할 것인지 설정
    + multi-gpu 사용 시 1000 * gpu 수 만큼 조절
  + `max_batches`
    + 언제까지 iteration 돌릴 것인지 설정
    + 보통 classes * 2000(또는 넉넉하게 4000) + 200
    +  뒤에 붙은 200은 전/후로 알맞은 가중치 얻기 위함
  + `policy` = (보통 steps)
  + `steps`
    + max_batches 사이즈(200을 더하지 않은 값)의 80%/90% 를 설정
  + `scales = 1.1`

+ network
  + `class = `(자신의 class 갯수로 수정)
  + `filters = `(classes + 4 + 1) * 3
    + 다른 해상도에 대한 정확도 높힐 경우
    + random (파일 맨 아래) = 1 
  + `Small Object` (416*416으로 Resizing 했을때 16*16보다 작은 경우)
  + `layers (720번째 줄) = -1, 11`
  + `stride (717번째 줄) = 4`
  + `flip (17번째 줄) = 0` : 좌우 구별 감지 원할 경우
  + `mask`
  + `anchors = `(수정 입력)
    + anchors 계산 :
  + `darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416` 

+ block
  + `Convolutional` : 
  + `Shortcut` : skip connection, ResNet에서 사용되는 것과 비슷 
    + `from = -3` : shortcut layer의 결과물이 이전 layer와 shortcut layer부터 뒤에서 3번째 layer의 feature map을 더해서 얻어지는 것 의미
  + `Upsample` : Bilinear upsampling을 사용해서 stride 값 만큼 이전 layer에서 feature map을 upsampling

  + `Route`
    + layers attribute(속성), 한개 혹은 2개의 값 갖음
    + layers 한개일 때, 그 값으로 layer의 feature map을 인덱싱 가능
      + 예) `layers = -4` : Route layer의 뒤에서 4번째 layer에서 feature map 출력
    + layers 두개일 때, 값으로 인덱싱 된 laye들의 feature map들을 concatenate 한 결과물 리턴
      + 예) `layers = -1, 61` : layer는 바로 이전 layer(-1)과 61번째 layer의 feature map 출력
  + yolo
    + Detection layer
    + anchors : 각각의 anchor들 표시
    + mask : 인덱싱(코드에 따라 0,1,2 or 6,7,8 번째 anchors들)에 해당하는 anchor들만 사용
    + 3개의 다른 scale detection layer을 갖고 있기 때문에 총 9개의 anchors
  + net
    + 네트워크 입력과 training parameters만 표기, layer 아님
    + YOLO forward pass에서 사용되지 않음. 하지만, forward pass시에 anchors를 수정하는데 사용되는 네트워크 입력 크기와 같은 정보들 제공



  <br>

## (obj).data
학습을 위한 내용 포함
+ `class = (자신의 class 갯수로 수정)`
+ train.txt, valid.txt, obj.names, weight 저장 폴더 경로
+ backup : iteration 거치면서 weight 가 저장 될 폴더 <br>


## (obj).names
Annotation 에 포함되어 있는, 검출 대상 목록 <br>


## train/valid.txt
학습 대상 또는 validation 이미지 경로가 담긴 리스트 <br>
train.txt 생성 코드
```python
import glob 
def file_path_save():
    filenames = []
    files = sorted(glob.glob("./obj/*.jpg"))
    for i in range(len(files)):
        f = open("./train.txt", 'a')
        f.write(files[i] + "\n")
if __name__ == '__main__':
    file_path_save()
```


## images (폴더)
학습시킬 이미지들
png or jpg
train/valid image 필요 <br>


## annotation (images 폴더에 함께 위치)
학습시킬 이미지들에 대한 주석
+ `[class id] [center_x] [center_y] [w] [h]`
각 이미지마다 주석들이 담긴 텍스트 파일 필요 <br>


## weight file
Pre-trained model 또는 기본적으로 darknet53.conv.74 등의 가중치 파일 <br>
Fine-tuning을 위해 맨 아래 레이어를 제거(AlexeyAB darknet에 내장)한 가중치 파일을 사용할 수도 있음 <br>
partial 을 이용하여 마지막 레이어 삭제 후 가중치 파일 생성 <br> <br>





