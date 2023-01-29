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

## 크롤링 성공한 과정

나는 google images download를 사용했는데, 이것은 개인이 구글에서 키워드에 따라 관련 이미지를 빠르게 대량으로 저장할 수 있도록 만든 파이썬 기반 프로그램이다.

하지만 구글에서는 시시각각으로 이미지를 호출하는 방식을 바꿔서 그에 맞춰 계속 진또배기 파이썬 코드를 계속 수정해줘야하는 어려움이 있다.

최근에 한번 구글이 이미지 호출 형식을 바꿨는지, 이상한 오류가 계속 발생하였고. 

![image](https://user-images.githubusercontent.com/117588181/215266215-a2be2956-be25-4729-b63a-a46c2a10320c.png)


이는 이미지에 대한 정보 중 url를 받아오지 못해서 생긴 오류이다.

실제로 이에 대한 문제를 제기한 사람이 github issue 부분에 있었고, 진짜 그 호출 방식을 뜯어보니 11번째에 url이 있어야하는데 비어있었다고 한다.

![image](https://user-images.githubusercontent.com/117588181/215266240-8e9a195c-7d78-491e-9588-e93733619d8c.png)


그래서 만약에 info가 비어있다면, data[11]을 찾아볼 것이 아니라, data[23]를 찾아보도록 수정했다.

그랬더니 정상적으로 이미지에 대한 정보를 받아와서 크롤링 후 저장하기를 시작했다.

![image](https://user-images.githubusercontent.com/117588181/215266258-83e6bfc5-397c-4a6a-ad4f-135407837a53.png)

매우 빠르게 100개의 스도쿠 사진을 저장했다.

## 크롤링 101개 이상 대용량으로 다운받는 법
- 100개씩 다운받기에는 데이터셋을 빠르게 만들기에 어려움이 있다.
- 난 일단 sudoku라는 키워드로 1000개를 다운받기 위해 시도했는데, 코드 자체가 100개를 기준으로 다운받는 방식이 분리되어있다. 
- 그리고 이참에 jpg 파일 형식만 다운받고 싶어서 기존에 테스트했던 사진 파일들도 모두 지웠다.
- 가장 먼저 chromedriver를 자신의 환경에 맞게 가져와서 exe 파일을 자신의 프로젝트 폴더 안에 넣는다.  
[chromedriver download](https://sites.google.com/chromium.org/driver/)
- 다음으로 searching.py 파일을 수정한다.
```python
from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"sudoku","limit":1000,"print_urls":True,"chromedriver":"E:/first/chromedriver.exe","format":"jpg"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images
```
- 최대한 절대주소로 적어주는 것이 나도 알아보기 쉽고 컴퓨터에게 정확히 알려주기 좋다고 한다.
- 경로 설정은 terminal창에 cd로도 가능하다.
- 실행해보면, 오류가 난다.
- 불과 3~4달 전에 파이썬 라이브러리가 개발자에 의해 수정되었다.  
[find_element_by_css_selector 참고 블로그](https://bskyvision.com/entry/python-selenium-%ED%81%AC%EB%A1%A4%EB%A7%81-findelementbycssselector-%EB%8D%94-%EC%9D%B4%EC%83%81-%EC%82%AC%EC%9A%A9-%EB%B6%88%EA%B0%80)
- 이 블로그를 참고하면, find_element_by_css_selector를 버리고 find_element를 사용하라고 했다.
- 그래서 google_images_download.py 파일에 들어가서 CTRL+F를 누르고 find_element를 찾는다.
- 정확히 3개가 나오는데, find_element_by_css_selector와 관련된 명령어를 모두 find_element로 수정한다. 
- 어떻게 업데이트 되었냐면, 수정할 부분을 매개변수로 넣는 방식으로 바뀌었다.
```python
try:
            browser.find_element(By.CSS_SELECTOR,"[aria-label='Accept all']").click()
            time.sleep(1)
        except selenium.common.exceptions.NoSuchElementException:
            pass

        print("Getting you a lot of images. This may take a few moments...")

        element = browser.find_element(By.TAG_NAME, "body")
        # Scroll down
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)

        try:
            browser.find_element(By.XPATH, '//input[@value="Show more results"]').click()
            for i in range(50):
                element.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.3)  # bot id protection
```
![image](https://user-images.githubusercontent.com/117588181/215311035-758f018d-6446-464a-ae2f-9cedc88cfa81.png)
- 참고해서 다 수정해주면 끝!
- 알아서 driver 잡고, 맞는 명령어로 크롤링까지 해준다.
