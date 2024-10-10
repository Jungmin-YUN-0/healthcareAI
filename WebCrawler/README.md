# WebCrawler for medical dataset crawling
- medical dataset을 크로링하고 csv로 저장
- Note : KSD 데이터셋은 selenium으로 크롤링 / KDCA 데이터셋은 beautifulsoup4로 크롤링 
  
## 크롤링 가능 데이터셋 항목
- MSD : (MSD 매뉴얼)[https://www.msdmanuals.com/ko-kr/home]
- KDCA : (국가건강정보포털)[https://www.nhis.or.kr/nhis/healthin/retrieveDiseVltGnlSymp.do]

## Usage - MSD
1. 필요한 라이브러리를 설치
```bash
pip install selenium tqdm pandas
```
2. 크롬 드라이버 설치
python script_name.py --url https://www.msdmanuals.com/ko-kr/home/symptoms --output symptoms_data.csv

3. 스크립트 실행
```bash
python run_MSD.py --url <크롤링할 URL> --output <출력 파일 경로>
```
예시 :
```bash
python script_name.py --url https://www.msdmanuals.com/ko-kr/home/symptoms --output symptoms_data.csv
```
- `--url`: MSD 매뉴얼에서 시작할 크롤링 URL (기본값은 "https://www.msdmanuals.com/ko-kr/home/symptoms", '증상' 페이지만 크롤링하는 경우)
- `--output`: 결과를 저장할 CSV 파일 경로

4. 데이터 형식
크롤링된 데이터는 아래와 같은 컬럼들로 구성된 CSV 파일로 저장

- `name`: 증상의 이름 (증상명 / 부제)
- `category`: '증상'으로 고정
- `explanation`: 증상에 대한 설명
- `causes`: 증상의 원인
- `evaluation`: 증상의 평가 방법
- `treatment`: 증상에 대한 치료법
- `keypoints`: 주요 요점
- `source`: 'MSD'로 고정
- `url`: 해당 증상 페이지의 URL

## Usage - KDCA
1. 필요한 라이브러리를 설치
```bash
pip install requests beautifulsoup4 tqdm pandas
```
2. 스크립트 실행
```bash
python run_KDCA.py
```
- 스크립트는 두 단계로 실행
    - Step 1: fetch_urls() 함수가 페이지 0부터 10까지 크롤링하여 건강 정보 페이지의 URL을 수집
    - Step 2: fetch_and_save_content() 함수가 수집된 URL을 통해 각 페이지에서 데이터를 추출하고 CSV 파일로 저장

3. 데이터 형식
크롤링된 데이터는 아래와 같은 컬럼들로 구성된 CSV 파일로 저장
- `Data`: 데이터 필드 이름
- `Description`: 데이터 필드의 설명

  
모든 데이터는 해당 건강 정보에 대한 별도의 CSV 파일로 저장되며, 저장 경로는 각 정보의 상위 카테고리 이름으로 된 폴더 안에 저장됩니다.
예를 들어, `진단 및 검사` 폴더 안에 `위내시경.csv` 형식으로 저장됩니다.
