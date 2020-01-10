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
`cfg`(configure) : 모델 튜닝, 데이터 적용 
- coco.data, yolov3.cfg(v3), yolo-obj.cfg(v2), rnn.cfg 등
`data` : 분석 대상
- coco.names : class 확인 가능 (80개)



## Makefile
콘솔 응용프로그램에서는 main 부터 찾는 것이 편함
makefile 확인하여 main 포함된 소스코드 확인 가능
```C++
GPU=0
CUDNN=0
OPENCV=0
OPENMP=0
DEBUG=0
 
ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?
 
# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52
 
VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/
 
CC=gcc
CPP=g++
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC
```

`GPU=1` : CUDA 설치 시 1, 미설치 0 <br>
`OPENCV=1` : CUDA 설치 시 1, 미설치 0 <br>

`VPATH=./src/:./examples` : 경로 설정, 소스파일 위치, examples <br>
`EXEC=darknet` : main source file (int main) 명 <br>
`OBJDIR=./obj/` : build 시 생성되는 object 파일들은 해당 경로에 출력됨 <br>

`CC` : C Compiler <br>
`NVCC=nvcc` : CUDA compiler <br>
`LDFLAGS` : linker option <br>
`CFLAGS` : compile option <br> <br>



## darknet/src/Darknet.c (Main source)

```C++
int main(int argc, char **argv)
{
    //test_resize("data/bad.jpg");
    //test_box();
    //test_convolutional_layer();
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif

    if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "yolo")){
        run_yolo(argc, argv);
    } else if (0 == strcmp(argv[1], "super")){
        run_super(argc, argv);
    } else if (0 == strcmp(argv[1], "lsd")){
        run_lsd(argc, argv);
    } else if (0 == strcmp(argv[1], "detector")){
        run_detector(argc, argv);
    } else if (0 == strcmp(argv[1], "detect")){
        float thresh = find_float_arg(argc, argv, "-thresh", .5);
        char *filename = (argc > 4) ? argv[4]: 0;
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        int fullscreen = find_arg(argc, argv, "-fullscreen");
        test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
    } else if (0 == strcmp(argv[1], "cifar")){
        run_cifar(argc, argv);
    } else if (0 == strcmp(argv[1], "go")){
        run_go(argc, argv);
    } else if (0 == strcmp(argv[1], "rnn")){
        run_char_rnn(argc, argv);
    } else if (0 == strcmp(argv[1], "coco")){
        run_coco(argc, argv);
    } else if (0 == strcmp(argv[1], "classify")){
        predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
    } else if (0 == strcmp(argv[1], "classifier")){
        run_classifier(argc, argv);
    } else if (0 == strcmp(argv[1], "regressor")){
        run_regressor(argc, argv);
    } else if (0 == strcmp(argv[1], "isegmenter")){
        run_isegmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "segmenter")){
        run_segmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "art")){
        run_art(argc, argv);
    } else if (0 == strcmp(argv[1], "tag")){
        run_tag(argc, argv);
    } else if (0 == strcmp(argv[1], "3d")){
        composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
    } else if (0 == strcmp(argv[1], "test")){
        test_resize(argv[2]);
    } else if (0 == strcmp(argv[1], "nightmare")){
        run_nightmare(argc, argv);
    } else if (0 == strcmp(argv[1], "rgbgr")){
        rgbgr_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "reset")){
        reset_normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "denormalize")){
        denormalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "statistics")){
        statistics_net(argv[2], argv[3]);
    } else if (0 == strcmp(argv[1], "normalize")){
        normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "rescale")){
        rescale_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "ops")){
        operations(argv[2]);
    } else if (0 == strcmp(argv[1], "speed")){
        speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
    } else if (0 == strcmp(argv[1], "oneoff")){
        oneoff(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "oneoff2")){
        oneoff2(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "print")){
        print_weights(argv[2], argv[3], atoi(argv[4]));
    } else if (0 == strcmp(argv[1], "partial")){
        partial(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "visualize")){
        visualize(argv[2], (argc > 3) ? argv[3] : 0);
    } else if (0 == strcmp(argv[1], "mkimg")){
        mkimg(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argv[7]);
    } else if (0 == strcmp(argv[1], "imtest")){
        test_resize(argv[2]);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
}
```
`detect` : 단일/멀티 이미지용, `test_detector` 함수 실행(detector.c 내부에 있음) <br>
*Option*<br>
`-ext_output` : 결과에 대한 상세 정보 표시<br>
`-i 1` : Use GPU 1<br>
`thresh 0.25 -dont_show -save_labels < list.txt` : 
- YOLO 임계치 조정
- 원래 25% 이상인 물체만 표시
- 해당 옵션을 이용하여 0% 이상인 모든 물체를 표시하게 하거나, 50% 이상의 물체만 탐지를 하는 등의 설정 가능
- List of Image에 대한 결과 저장 <br> <br>

`detector` : 동영상용 <br> <br>



## detector.c
`draw_detections`: detection 결과 표현, 갯수 출력, 인식 물체 이름 등 정의






## yolo.c
```cpp
char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
char *train_images = "/data/voc/train.txt";
char *backup_directory = "/home/pjreddie/backup/";
```


`char *voc_names[]` : 클래스 이름 설정 변수 <br>
`char *train_images` : 학습할 image들의 list.txt파일 위치 <br>
`char *backup_directory` : 학습을 하면서 중간결과들을 저장해놓는 폴더 위치, 최종 가중치 파일 저장 위치 동일 <br><br>



## .cfg
```sh

```
`batch = 64` <br>
`subdivision = 8` Out of memory 오류 시, subdivision 을 16, 32, 64 등으로 증가시킴 <br>
`height/width = `(32의 배수, 608 or 832, 클수록 정확도 향상) <br>
`class = (자신의 class 갯수로 수정)` <br>
`filters (class 위에 있음) = (classes + 5) * 3` <br>
- 다른 해상도에 대한 정확도 높힐 경우 <br>
- random (파일 맨 아래) = 1  <br>
`Small Object(416*416으로 Resizing 했을때 16*16보다 작은 경우)` <br>
`layers (720번째 줄) = -1, 11` <br>
`stride (717번째 줄) = 4` <br>
`anchors = (수정 입력)` <br>
- anchors 계산 : <br>
`darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416` <br>
`flip (17번째 줄) = 0` : 좌우 구별 감지 원할 경우
 <br> <br>



