# [Yolo v3] Object Detection

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
콘솔 응용프로그램에서는 main 부터 찾는 것이 편함 <br>
makefile 확인하여 main 포함된 소스코드 확인 가능 <br>
+ `GPU=1` : CUDA 설치 시 1, 미설치 0 <br>
+ `OPENCV=1` : CUDA 설치 시 1, 미설치 0 <br>

+ `VPATH=./src/:./examples` : 경로 설정, 소스파일 위치, examples <br>
+ `EXEC=darknet` : main source file (int main) 명 <br>
+ `OBJDIR=./obj/` : build 시 생성되는 object 파일들은 해당 경로에 출력됨 <br>

+ `CC` : C Compiler <br>
+ `NVCC=nvcc` : CUDA compiler <br>
+ `LDFLAGS` : linker option <br>
+ `CFLAGS` : compile option <br> <br>



## darknet/src/Darknet.c (Main source)
+ `detect` : 단일/멀티 이미지용, `test_detector` 함수 실행(detector.c 내부에 있음) <br>
  + *Option*<br>
    + `-ext_output` : 결과에 대한 상세 정보 표시<br>
    + `-i 1` : Use GPU 1 <br>
    + `thresh 0.25 -dont_show -save_labels < list.txt` : 
      + YOLO 임계치 조정 <br>
      + 원래 25% 이상인 물체만 표시 <br>
      + 해당 옵션을 이용하여 0% 이상인 모든 물체를 표시하게 하거나, 50% 이상의 물체만 탐지를 하는 등의 설정 가능 <br>
    + List of Image에 대한 결과 저장 <br> <br>

+ `detector` : 동영상용 <br> <br>



## detector.c
`draw_detections`: detection 결과 표현, 갯수 출력, 인식 물체 이름 등 정의




## yolo.c
`char *voc_names[]` : 클래스 이름 설정 변수 <br>
`char *train_images` : 학습할 image들의 list.txt파일 위치 <br>
`char *backup_directory` : 학습을 하면서 중간결과들을 저장해놓는 폴더 위치, 최종 가중치 파일 저장 위치 동일 <br><br>



## (obj).cfg
모델 구조 및 train과 관련된 설정 포함

[data feed] <br>
`batch = 64` <br>
`subdivision = 8` <br>
    - Cuda memory 관련 <br>
    - 배치 사이즈를 얼마나 쪼개서 학습할 것인지에 대한 설정
    - Out of memory 오류 시, subdivision 을 16, 32, 64 등으로 증가시킴 <br>
`height/width = 416`(32의 배수, 608 or 832 추천, 클수록 정확도 향상) <br>

[augmentation]
`angle`: 이미지 회전 정도 설정, 보통은 0, 경우에 따라 90도 가능 <br>
`saturation` : 채도 추가 원할 때 설정 <br>
`expose` : 노출 값 추가 원할 때 설정 <br>
`hue` : 색상 변경 원할 때 설정 <br>

[training] <br>
`learning rate = 0.001`  <br>
    - muti-gpu 사용 시 학습율 0.001/gpu 수만큼 조절하기도 함 <br>
`burn_in = 1000` <br>
    - multi-gpu 사용 시 몇 만큼의 iteration 지날 때 마다 학습률을 조장 할 것인지 설정 <br>
    - multi-gpu 사용 시 1000 * gpu 수 만큼 조절 <br>
`max_batches` <br>
    - 언제까지 iteration 돌릴 것인지 설정 <br>
    - 보통 classes * 2000(또는 넉넉하게 4000) + 200 <br>
    - 뒤에 붙은 200은 전/후로 알맞은 가중치 얻기 위함 <br>
`policy` = (보통 steps) <br>
`steps` <br>
    - max_batches 사이즈(200을 더하지 않은 값)의 80%/90% 를 설정 <br>
`scales = 1.1` <br>

[network] <br>
`class = `(자신의 class 갯수로 수정) <br>
`filters = `(classes + 4 + 1) * 3 <br>
- 다른 해상도에 대한 정확도 높힐 경우 <br>
- random (파일 맨 아래) = 1  <br>
`Small Object` (416*416으로 Resizing 했을때 16*16보다 작은 경우) <br>
`layers (720번째 줄) = -1, 11` <br>
`stride (717번째 줄) = 4` <br>
`flip (17번째 줄) = 0` : 좌우 구별 감지 원할 경우 <br>
`mask` <br>
`anchors = `(수정 입력) <br>
    - anchors 계산 : <br>
`darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416` <br> <br>

## (obj).data
학습을 위한 내용 포함
- `class = (자신의 class 갯수로 수정)` <br>
- train.txt, valid.txt, obj.names, weight 저장 폴더 경로 <br>
- backup : iteration 거치면서 weight 가 저장 될 폴더 <br> <br>


## (obj).names
Annotation 에 포함되어 있는, 검출 대상 목록 <br> <br>


## train/valid.txt
학습 대상 또는 validation 이미지 경로가 담긴 리스트 <br> <br>
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
train/valid image 필요 <br> <br>


## annotation (images 폴더에 함께 위치)
학습시킬 이미지들에 대한 주석
`[class id] [center_x] [center_y] [w] [h]` <br>
각 이미지마다 주석들이 담긴 텍스트 파일 필요 <br> <br>



## weight file
Pre-trained model 또는 기본적으로 darknet53.conv.74 등의 가중치 파일
Fine-tuning을 위해 맨 아래 레이어를 제거(AlexeyAB darknet에 내장)한 가중치 파일을 사용할 수도 있음
partial 을 이용하여 마지막 레이어 삭제 후 가중치 파일 생성 <br> <br>





