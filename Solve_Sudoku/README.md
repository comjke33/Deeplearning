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

원래,

1. 이미지의 여백 제거 → 되었다고 가정
2. 이미지에서 테이블 추출
    1. openCV를 사용한 테이블 감지 및 셀 인식
    2. 적절한 행과 열에 대한 셀 할당
    3. 셀 추출