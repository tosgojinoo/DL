# [Yolo] Darknet

## 참고
+ [Yolo Darknet page](https://pjreddie.com/darknet/yolo/)
+ yolo v3 저자 [pjreddie](https://github.com/pjreddie/darknet)
+ 부가기능 사용시 [AlexeyAB](https://github.com/AlexeyAB/darknet)
<br>

## Yolo
+ You Only Look Once
+ Convolutional layers 들만 사용하여 fully convolutional network(FCN) 구성
    + FCN 이므로 입력이미지 크기가 속도에 영향 주지 않음
    + 하지만, 문제 발생 방지를 위해 일정한 입력 크기 유지 필요
+ Skip connections 와 upsampling* layer 들을 포함해 총 75개의 convolutional layer 들로 구성
+ Pooling 은 사용되지 않음
+ feature map 을 downsample 하기 위해서 두개의 stride를 가진 convolutional layer 사용
    + pooling 으로 인해 low-lever feature 들이 자주 loss 되는 것 방지
+ batch 로 이미지 처리시, 모든 이미지들의 높이/너비 고정 필요
    + GPU로 병렬처리 가능, 속도 향상
    + 여러 이미지를 하나의 큰 batch로 연결시 필요 (= 여러개 Pytorch tensor들을 하나로 연결)
+ stride -> 이미지 downsampling
<br>

## Anchor Boxes
+ Bounding box의 높이/너비는 training시에 불안정한 gradient를 발생시킴
+ 그래서, 대부분 log-space transforms 예측, 또는 anchors 라고 불리는 미리 정의된 bounding box 사용
+ transforms -> anchor box 적용 -> 예측값 산출
+ (YOLOv3) 3개 anchors -> 각 cell마다 3 개의 bounding box 예측
+ 예) 개를 detect하는 것에 책임을 지게 될 bounding box는 ground truth box와 함께 가장 높은 IoU를 가진 anchor가 될 것임
<br>

## Pre-requirement
+ OpenCV 2.4 이상
+ CUDA
+ cuDNN 
<br>

## 실행
### 기본
```sh
git clone https://github.com/pjreddie/darknet
cd darknet
make
```
<br>

### weight 받아올 경우
```sh
wget https://pjreddie.com/media/files/yolov2.weights
./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights data/dog.jpg
또는
./darknet detect cfg/yolov2.cfg yolov2.weights data/dog.jpg -thresh 0
```
<br>

### 폴더 구조
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
+ Makefile
	+ CUDA 설치 시, GPU=1
	+ OpenCV 설치 시, OPENCV=1
	+ 둘다 미설치 시, 0
	+ 추가
		+ `NVCC=nvcc`          -> CUDA compiler
		+ `VPATH=./src/`       -> 소스코드 위치
		+ `EXEC`               -> 실행 파일 명
		+ `CC`                 -> C Compiler
		+ `LDFLAGS`            -> linker option
		+ `CFLAGS`             -> compile option
+ 설정 완료 후, 터미널창에 make 명령어 입력하여 코드 Complile
+ 완료되면, darknet 실행파일 생성

#### 참고) make & makefile
+ shell 에서 컴파일 시, `make` 명령어로 `makefile` 컴파일 가능 
+ `make`는 파일 간의 종속관계를 파악하여 `makefile`에 적힌 대로 컴파일러에 명령, shell 명령이 순차적으로 실행될 수 있게 함 
+ 장점은,
	+ 각 파일에 대한 반복적 명령의 자동화로 인한 시간 절약
	+ 프로그램 종속 구조 빠르게 파악하여, 관리 용이
	+ 단순 반복 작업, 재작성 최소화

+ `make` : 파일 관리 유틸리티 
+ `makefile` : 기술파일(script) 
<br>

### Train
+ Yolo Darknet 폴더 안 scr/yolo.c 파일 내용
```sh
char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
char *train_images = "/data/voc/train.txt";
char *backup_directory = "/home/pjreddie/backup/";
```
+
	+ `char *voc_names[]` : 클래스 이름 설정 변수 
	+ `char *train_images` : 학습할 image들의 list.txt파일 위치 
	+ `char *backup_directory` : 학습을 하면서 중간결과들을 저장해놓는 폴더 위치, 최종 가중치 파일 저장 위치 동일 

+ 용도에 맞게 위 내용 변경
+ 딥러닝 모델 생성(cfg 파일) 또는 수정
	+ batch = 64
	+ subdivision = 8
		+ Out of memory 오류 시, subdivision 을 16, 32, 64 등으로 증가시킴
	+ height/width = (32의 배수, 608 or 832, 클수록 정확도 향상)
	+ class = (자신의 class 갯수로 수정)
	+ filters (class 위에 있음) = (classes + 5) * 3
	+ 다른 해상도에 대한 정확도 높힐 경우
		+ random (파일 맨 아래) = 1 
	+ Small Object(416*416으로 Resizing 했을때 16*16보다 작은 경우)
		+ layers (720번째 줄) = -1, 11
		+ stride (717번째 줄) = 4
	+ anchors = (수정 입력)
		+ anchors 계산 :
`darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`

	+ 좌우 구별 감지 원할 경우
		+ flip (17번째 줄) = 0

+ data 파일 생성
```sh
classes = 3 
train = EXAMPLE1/train.txt 
valid = EXAMPLE1/test.txt 
names = EXAMPLE1/obj.names 
backup = EXAMPLE1/backup/ # 중간 weights를 저장하는 경로
```
+ names 파일 생성
	+ class 0, 1, 2 ... 작성
```sh
bird
dog
cat
```
+ Training/Testing에 사용할 이미지 파일 저장
	+ 경로 : `build\darknet\x64\EXAMPLE1 (for win)`
	+ Bounding Box 처리 완료된 이미지 사용
	+ [Marking 툴](https://github.com/AlexeyAB/Yolo_mark)
+ 이미지 리스트의 상대 경로 적힌 txt 파일 생성(train.txt, test.txt)
```sh
EXAMPLE1/img1.jpg
EXAMPLE1/img2.jpg
EXAMPLE1/img3.jpg
```
+ dark 폴더에서 make 실행
+ 다음 명령어 입력
```sh
./darknet detector train .data .cfg .weights
./darknet detector train cfg/yolo.cfg (pre-trained model)
```
+ Pre-trained Model 없을 시, 가중치는 자체적으로 초기화한 가중치 값 사용
	+ training 시 Loss-window 없애려면 <code>-dont_show</code> 옵션 설정
+ training 후 성능 확인
```sh
./darknet detector map .data .cfg .weights
```
+ mAP-chart (평균정밀도 평균, Mean Average Precision)
```sh
./darknet detector train .data .cfg .weights -map
```
<br>

### Test
+ obj_test.cfg 생성
	+ obj.cfg 를 변형
	+ batch 및 subdivisions 사이즈 1로 조정
	+ 결과 향상 위해 width/height 를 608 로 변경

+ 명령어 구성
```sh
./(실행파일) (Darknet에서 지원하는 딥러닝 아키텍쳐 종류) (사용할 함수 이름) (설정파일) (가중치 파일, weights) (추가옵션)
```
=> test.sh 생성됨

+ test.sh 실행
```sh
$ ./obj_test.sh
```

+ Image
(이미지 파일 실행 후) 
```sh
./darknet detector test .data .cfg .weights -thresh THRESH OPTION
or
./darknet detector test cfg/yolo.cfg yolo.weights data/dog.jpg
```
+ 
	+ *Option*
		+ `-ext_output` : Output coordinates
		+ `-i 1` : Use GPU 1
		+ `thresh 0.25 -dont_show -save_labels < list.txt` : List of Image에 대한 결과 저장


+ Video
(동영상 파일 실행)
```sh
./darknet detector demo .data .cfg .weights .videofile OPTION
or
./darknet detector demo cfg/yolo.cfg yolo.weights -c <number> : 카메라 index number
or
./darknet detector demo cfg/yolo.cfg yolo.wegiths test.mp4 : 동영상에 대한 테스트
```
+ 
	+ *Option*
		+ `-c 0` : WebCam 0
		+ `http://주소` : Net-videocam
		+ `-out_filename OUT.videofile` : 결과 저장


+ Check accuracy mAP@IoU=75
```sh
./darknet detector map .data .cfg .weights -iou_thresh 0.75
```
<br>

### log
학습시 생성된 log 확인 
[마지막 부분]
+ Region 82
	+ 가장 큰 Mask, Prediction Scale 을 이용하는 레이어이지만 작은 객체를 예측 할 수 있음
+ Region 94
	+ 중간 단계 Mask 
+ Region 106
	+ 가장 작은 Mask, Prediction Scale 을 이용하는 레이어이지만 마스크가 작을 수록 큰 객체 예측 가능
+ Avg IOU 
	+ 현재의 subdivision에서 이미지의 평균 IoU
	+ 실제 GT와 예측된 bounding box의 교차율을 뜻함
	+ 1에 가까울 수록 좋음
+ Class : 1에 가까운 값일 수록 학습이 잘 되고 있다는 것
+ No Obj : 값이 0이 아닌 작은 값이어야 함
+ .5R : recall/conut 
+ .75R : 0.000000
+ count : 현재 subdivision 이미지들에서 positive sample 들을 포함한 이미지의 수 

[중간 부분]
+ loss avg 확인 필요(3열)
	+ iteration 이 증가할 수록 loss avg 값 감소
	+ 중간에 계속 증가할 경우 문제됨
	+ 많은 데이터의 경우 loss 값이 3.xxx 까지 떨어지며 더 이상 감소하지 않는 현상 발생
	+ 적은 데이터의 경우 loss 값이 0.6 정도나 그 이하까지 감소 가능
	+ 더 이상 Loss 값이 떨어지지 않을 때 학습 정지
	+ 권장 iteration (= 클래스 수 * 2000) 만큼 학습 추천
<br>








