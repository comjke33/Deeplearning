## 내가 원하는 것은

1. 웹캠으로 스도쿠를 찍는다.
2. 각 모서리를 감지해서 스도쿠의 table 형식을 sementic segmentation deep learning으로 감지한다.
3. 스도쿠에 적힌 숫자를 위치에 맞게 배열에 저장하면 이중 for문을 돌려서 81개의 숫자가 행, 열에 관계없이 정확히 맞는지 검사한다.
4. 맞으면 O를, 틀리면 X를 text로 보여준다.

## Sementic Segmentation에 대하여

[즐기며 사는 남자 : 네이버 블로그](https://blog.naver.com/mincheol9166/221740224096)

매우 설명이 잘 되어있고, 내 수준에서 이해하기 쉽게 설명되어있었다.

## 1. 웹캠으로 스도쿠를 찍으면,

프레임을 계속 나눠서 특정 한 순간이라도 가장 큰 표 모양을 감지할 수 있어야한다.

그 표의 끝점 위치를 파악해야한다.

- 어렵게 안하는 방법도 많은 것 같다.

[Image-Table-OCR - 표 이미지를 CSV로 변환 | GeekNews](https://news.hada.io/topic?id=3874)

→ 그냥 바로 이미지를 인식해서 배열을 파악해 그리드로 보여주는 방식도 있다.

- 일단 가장 먼저 입력 데이터인 스도쿠 이미지가 정확히 잘라졌을 때를 가정해서 제작해보기로 하였다.
- 스도쿠 이미지의 여백, 사이즈 뭐든것이 웹캠에 정확히 주어졌을 때(ㅎㅎ)
- 나중에 참고1

[OpenCV - 18. 경계 검출 (미분 필터, 로버츠 교차 필터, 프리윗 필터, 소벨 필터, 샤르 필터, 라플라시안 필터, 캐니 엣지)](https://bkshin.tistory.com/entry/OpenCV-18-%EA%B2%BD%EA%B3%84-%EA%B2%80%EC%B6%9C-%EB%AF%B8%EB%B6%84-%ED%95%84%ED%84%B0-%EB%A1%9C%EB%B2%84%EC%B8%A0-%EA%B5%90%EC%B0%A8-%ED%95%84%ED%84%B0-%ED%94%84%EB%A6%AC%EC%9C%97-%ED%95%84%ED%84%B0-%EC%86%8C%EB%B2%A8-%ED%95%84%ED%84%B0-%EC%83%A4%EB%A5%B4-%ED%95%84%ED%84%B0-%EB%9D%BC%ED%94%8C%EB%9D%BC%EC%8B%9C%EC%95%88-%ED%95%84%ED%84%B0-%EC%BA%90%EB%8B%88-%EC%97%A3%EC%A7%80)


1. 이미지의 여백 제거 → 되었다고 가정
2. 이미지에서 테이블 추출
    1. openCV를 사용한 테이블 감지 및 셀 인식
    2. 적절한 행과 열에 대한 셀 할당
    3. 셀 추출
3. 셀에 있는 숫자 인식
4. 배열 제작 
5-1. 정답인지, 오답인지 체크
5-2. 문제 풀이

### 1. Data 불러오기

- openCV가 무엇인지 대충 알게 되었다.
- 먼저 dataset을 불러오기 위해, 디렉토리를 불러왔다.

```python
data_classes=len(os.listdir("E:/sudoku/sudoku"))
```

- 그리고 print(data_classes)하면 내가 만든 사진 갯수인 323이 나온다.
- 사진 파일을 잘 다루기 위해서 파일들의 이름에 일괄적으로 숫자를 붙여야했다.
- 그건 DarkNamer 프로그램을 다운받아서 자릿수 채움 없이 0부터 322까지 이름 붙였다.
- cv2에 사진이 잘 들어가고 있는지 확인차, 이 코드를 작성했다.

```python
data_classes=len(os.listdir("E:/sudoku/sudoku"))
pic = cv2.imread("E:/sudoku/sudoku/0.jpg")
cv2.imshow('image', pic)
```

- imread와 imshow를 이용하면 내가 고른 첫번째 사진에 대해 창으로 띄워 보여줄 수 있다.
- 근데, 이렇게 테스트 하면 내가 볼 겨를도 없이 창이 나타났다가 바로 사라진다.

```python
data_classes=len(os.listdir("E:/sudoku/sudoku"))
pic = cv2.imread("E:/sudoku/sudoku/0.jpg")
cv2.imshow('image', pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- 마지막에 이 2줄을 추가해주면 내가 아무 버튼이나 누르기 전까지 창이 꺼지지 않는다.
- 이제 사진을 opencv에서 불러오는 것을 성공했다..

```python
data_classes=len(os.listdir("E:/sudoku/sudoku"))
data_X = []
data_Y = []
for i in range(0,data_classes):
    pic = cv2.imread("E:/sudoku/sudoku" +"/"+str(i)+".jpg")
    pic = cv2.resize(pic,(32,32)) 
    data_X.append(pic)
    data_Y.append(i)
    if len(data_X)==len(data_Y):
        print("Total Datapoints= ", len(data_X))
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e1cd3862-ecce-4f73-a5fc-354162c33267/Untitled.png)

- 모든 사진을 불러와서 사이즈를 32*32로 조정하는 것을 성공했다.

---

### Test, Train, Validation 데이터 분리

```python
train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.05)
train_X, validation_X, train_Y, validation_Y = train_test_split(train_X, train_Y, test_size=0.2)
print("Training Set Shape = ",train_X.shape)
print("Validation Set Shape = ",validation_X.shape)
print("Test Set Shape = ",test_X.shape)
```

- dataset를 train 데이터와 test 데이터 그리고 Validation 데이터로 나눈다.
- 먼저 Test 데이터로 5%의 데이터, Train 데이터로 95%의 데이터를 떼어주었다.
- 그리고 Train 데이터에서 다시 Train 데이터 80% 그리고 validation 데이터로 20%를 떼어주었다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0062d50f-48b0-4b07-8211-07fb2bef7968/Untitled.png)

- 데이터가 너무 적어서… 어쩔 수 없다. 계속 틈날 때마다 사진 찍어서 추가해야겠다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/70a361fe-667e-4e3d-aa29-3ce583acdd21/Untitled.png)
  
