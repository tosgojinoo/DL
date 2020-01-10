# [Yolo] labeling

## 개요
- 직접 데이터 학습을 위한 작업
- 이미지는 JPG로 준비 (필요시, 알집으로 한번에 이미지 포맷 변환 가능)

## [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark)(c++)
- ```git clone https://github.com/AlexeyAB/Yolo_mark```
- `yolo_mark/x64/release/data` 이동
	- `/img` 에 기존 이미지 파일 삭제 후 본인 이미지 삽입
	- `vi obj.data`
		- classes = (원하는 분류수)
		- train, valid, names, backup 경로 확인
	- `vi obj.names`
		- 분류 class 이름 붙이기
- `yolo_mark`이동
	- `./linux_mark.sh` (실행)
- bounding box 직접 그려넣어 labeling
	- jpg 옆에 txt 파일 생성(box 좌표값)
- `yolo_mark/x64/release` 이동
	- `vi yolo-obj.cfg` (실행)
		- [convolutional] filters = 5 * (classes + 5)
		- [region] classes = (원하는 분류수)
- convolutional layer 설치 (darknet 디렉토리 안) : http://pjreddie.com/media/files/darknet19_448.conv.23 (-> weight)
- `yolo_mark/x64/release/`yolo-obj.cfg -> darknet 디렉토리로 이동
- `yolo_mark/x64/release/data/`image, obj.names, obj.data, train.txt -> darknet/data 디렉토리로 이동

## [LabelImg](https://github.com/tzutalin/labelImg)(python)

## [OpenLabeling](https://github.com/Cartucho/OpenLabeling)(python)

## [Darkmark](https://www.ccoderun.ca/darkmark/)(c++)

## [Cvat](https://github.com/opencv/cvat)(javascript)


## Use
- darknet 실행 파일 생성 경로에 함께 위치

