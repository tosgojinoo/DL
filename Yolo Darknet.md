# Yolo Darknet

## 구성 및 설치
- 참고
[Yolo Darknet page](https://pjreddie.com/darknet/yolo/)


### Pre-requirement
- OpenCV
- CUDA

### Install
- [소스 확인](https://github.com/pjreddie/darknet)
- 폴더 구조
```sh
darknet
  |
  |- cfg/
  |- data/
  |- LICENSE
  |- Makefile
  |- obj/
  |- README.md
  |- results/
  |- scripts/ 
  |- src/
```
- Makefile
	- CUDA 설치 시, GPU=1
	- OpenCV 설치 시, OPENCV=1
	- 둘다 미설치 시, 0
	- 추가
		- `NVCC=nvcc`          -> CUDA compiler
		- `VPATH=./src/`       -> 소스코드 위치
		- `EXEC`               -> 실행 파일 명
		- `CC`                 -> C Compiler
		- `LDFLAGS`            -> linker option
		- `CFLAGS`             -> compile option
- 설정 완료 후, 터미널창에 make 명령어 입력하여 코드 Complile
- 완료되면, darknet 실행파일 생성

#### 참고) make & makefile
shell 에서 컴파일 시, `make` 명령어로 `makefile` 컴파일 가능 <br>
`make`는 파일 간의 종속관계를 파악하여 `makefile`에 적힌 대로 컴파일러에 명령, shell 명령이 순차적으로 실행될 수 있게 함 <br>
장점
- 각 파일에 대한 반복적 명령의 자동화로 인한 시간 절약
- 프로그램 종속 구조 빠르게 파악하여, 관리 용이
- 단순 반복 작업, 재작성 최소화

`make` : 파일 관리 유틸리티
`makefile` : 기술파일(script)


### Test
- 명령어 구성
```sh
./(실행파일) (Darknet에서 지원하는 딥러닝 아키텍쳐 종류) (사용할 함수 이름) (설정파일) (가중치 파일, weights) (추가옵션)
```

- 단일 이미지 테스트
```sh
./darknet yolo test cfg/yolo.cfg yolo.weights data/dog.jpg
```

- 카메라 스트리밍 영상 혹은 동영상에 대한 테스트
```sh
./darknet yolo demo cfg/yolo.cfg yolo.weights -c<number> : 카메라 index number
./darknet yolo demo cfg/yolo.cfg yolo.wegiths test.mp4 : 동영상에 대한 테스트
```

### Train
- Yolo Darknet 폴더 안 scr/yolo.c 파일 내용
```sh
char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
char *train_images = "/data/voc/train.txt";
char *backup_directory = "/home/pjreddie/backup/";
```

`char *voc_names[]` : 클래스 이름 설정 변수 <br>
`char *train_images` : 학습할 image들의 list.txt파일 위치 <br>
`char *backup_directory` : 학습을 하면서 중간결과들을 저장해놓는 폴더 위치, 최종 가중치 파일 저장 위치 동일 <br>

- 용도에 맞게 위 내용 변경
- 딥러닝 모델 생성(cfg 파일)
- dark 폴더에서 make 실행
- 다음 명령어 입력
```sh
./darknet yolo train cfg/yolo.cfg (pre-trained model)
```
- Pre-trained Model 없을 시, 가중치는 자체적으로 초기화한 가중치 값 사용





















