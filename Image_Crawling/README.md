## 크롤링 성공한 과정

나는 google images download를 사용했는데, 이것은 개인이 구글에서 키워드에 따라 관련 이미지를 빠르게 대량으로 저장할 수 있도록 만든 파이썬 기반 프로그램이다.

하지만 구글에서는 시시각각으로 이미지를 호출하는 방식을 바꿔서 그에 맞춰 계속 진또배기 파이썬 코드를 계속 수정해줘야하는 어려움이 있다.

최근에 한번 구글이 이미지 호출 형식을 바꿨는지, 이상한 오류가 계속 발생하였고. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/03cc15c6-7f46-4972-b8b8-7dfd712cda8c/Untitled.png)

이는 이미지에 대한 정보 중 url를 받아오지 못해서 생긴 오류이다.

실제로 이에 대한 문제를 제기한 사람이 github issue 부분에 있었고, 진짜 그 호출 방식을 뜯어보니 11번째에 url이 있어야하는데 비어있었다고 한다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/97834006-943f-443d-bafa-325ce98b13c2/Untitled.png)

그래서 만약에 info가 비어있다면, data[11]을 찾아볼 것이 아니라, data[23]를 찾아보도록 수정했다.

그랬더니 정상적으로 이미지에 대한 정보를 받아와서 크롤링 후 저장하기를 시작했다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fb1c5edb-d8a5-434b-94e4-3891fd71675f/Untitled.png)

매우 빠르게 100개의 스도쿠 사진을 저장했다.

## google-images-download 이용해서 크롤링 하는 방법

- 대용량 이미지를 다운받는 것이 대부분이니, 충분한 크기의 저장소를 확보하자.
- 폴더 하나를 만든다.
- VScode에서 그 폴더를 연다.
- 터미널을 열어서 그 폴더 안에서 이 명령어를 친다.
```python
pip install git+https://github.com/Joeclinton1/google-images-download.git
```
- 그 다운받은 폴더 내에 같은 이름(google_images_download)의 폴더가 또 있을 것이다.
- 하지만 그 폴더가 아닌, 제일 큰 폴더 안에 파이썬 파일을 하나 만든다.
```python
from google_images_download import google_images_download 
response = google_images_download.googleimagesdownload()  
arguments = {"keywords":"단어","limit": 개수,"print_urls":True}  
paths = response.download(arguments)  
print(paths)  
```
- 이렇게 적고 디버깅을 해본다. -> 100% 오류난다
- 오류 중에 링크가 있다. ctrl를 누르고 들어가면, 작은 google_images_download 파일 안으로 들어가진다.
- 그리고 그 파일의 409번째 줄에 가보면 if문이 있는데, 그 안의 info = data[11]을 info = data[23]으로 바꿔준다.
- 다시 디버깅 없이 실행. -> 성공
