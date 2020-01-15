# [Yolo] with pytorch

## 구성
+ `Darknet.py` : YOLO 네트워크 코드 작성
    + `parse_cfg`
        + 목적 : cfg를 파싱을 해서 모든 block을 dict 형식으로 저장
        + 인자 :  Configuration 파일
        + 리턴 : Blocks 리스트 
        + block들의 attribute, value 들을 dictionary에 key-value 형식으로 저장
        + 각 block은 neural network를 어떻게 빌드하는지에 대해 표기
        
```python
file = open(cfgfile, 'r')
lines = file.read().split('\n')                        # list로 lines에 저장한다
lines = [x for x in lines if len(x) > 0]               # 비어있는 line들 제거한다 
lines = [x for x in lines if x[0] != '#']              # 주석들 제거한다
lines = [x.rstrip().lstrip() for x in lines]           # whitespaces 제거한다

block = {}
blocks = []

for line in lines:
    if line[0] == "[":               # 새로운 block의 시작
        if len(block) != 0:          # block이 비어있지 않다면, 이전 block뒤에 값을 추가한다
            blocks.append(block)     # blcok list에 추가한다 
            block = {}               # block 다시 비운다
        block["type"] = line[1:-1].rstrip()     
    else:
        ckey,value = line.split("=") 
        block[key.rstrip()] = value.lstrip()
blocks.append(block)

return blocks
```
<br>


 
    
    
    
+ `util.py` : helper 함수 구현
+ `*.cfg` : 네트워크 빌드

