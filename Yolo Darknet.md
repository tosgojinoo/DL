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
'''sh
darknet
  |
  |- cfg /
  |- data/
  |-LICENSE
  |-Makefile
  |-obj/
  |-README.md
  |-results/
  |-scripts/ 
  |-src/
'''
- Makefile
	- CUDA 설치시, GPU=1
	- OpenCV 설치시, OPENCV=1
	- 둘다 미설치시, 0
	- 추가
		- NVCC=nvcc          -> CUDA compiler
		- VPATH=./src/       -> 소스코드 위치
		- EXEC               -> 실행 파일 명
		- CC                 -> C Compiler
		- LDFLAGS            -> linker option
		- CFLAGS             -> compile option