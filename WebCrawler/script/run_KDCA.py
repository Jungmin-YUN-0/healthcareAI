import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm
import time
import os

def fetch_urls():
    """
    Fetch URLs from a fixed range of pages.
    """
    total_urls = []
    session_id = 0

    for page in tqdm(range(0, 11)):  # 페이지 0부터 10까지 (총 11페이지)
        if page == 0:
            data = {"lclasSn": session_id}
        else:
            data = {
                "lclasSn": session_id,
                "initial": None,
                "age": None,
                "gender": None,
                "bdySystem": None,
                "bdySystemNm": None,
                "dissstle": None,
                "cntnts_sn": None,
                "cancerSeq": None,
                "cancerName": None,
                "searchTy": "U",
                "pageIndex": page
            }

        response = requests.get(
            'https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoMain.do', 
            params=data
        )
        time.sleep(5)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            urls_on_page = [
                link['href'].split("'")[1]
                for link in soup.select('div.hd-indexbox ul li a')
                if 'java' in link['href']
            ]
            total_urls.extend(urls_on_page)
        else:
            print(f"Failed to fetch page {page}. Status code: {response.status_code}")

    return total_urls, response.cookies


def fetch_and_save_content(url_list, cookies):
    """
    Fetch content data from the list of URLs and save it to CSV files.
    """
    for url in tqdm(url_list[0:]):  # 인덱스를 0부터 시작하여 모든 URL 처리
        data = {
            "lclasSn": None,
            "initial": None,
            "age": None,
            "gender": None,
            "bdySystem": None,
            "bdySystemNm": None,
            "dissstle": None,
            "cntnts_sn": url,
            "cancerSeq": None,
            "cancerName": None,
            "searchTy": "U"
        }

        response = requests.post(
            'https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoView.do', 
            data=data, 
            cookies=cookies
        )

        try:
            time.sleep(1)
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text().split('본 공공저작물은')[0].split('정보신청 :')[1]

            matching_lines = [
                line.strip()
                for line in text_content.split('\n\n')
                if re.match(r'^\s*\n.{0,10}\n.*', line)
            ]

            content_data = pd.DataFrame(
                [item.split(':')[:2] for item in text_content.split('\n\n')[2].split('•')[1:]] +
                [line.split('\n') for line in matching_lines if len(line.split('\n')) == 2]
            )

            superclass_name = content_data.iloc[0, 1].strip()
            content_name = content_data.iloc[1, 1].strip()

            if not os.path.exists(superclass_name):
                os.makedirs(superclass_name)
                print(f"Folder '{superclass_name}' created.")
            else:
                print(f"Folder '{superclass_name}' already exists.")

            content_data.columns = ['Data', 'Description']
            content_data.set_index('Data')

            file_path = os.path.join(superclass_name, f'{content_name}.csv')
            content_data.to_csv(file_path.replace('/', ''), index=False)

            print(f"File '{file_path}' saved as CSV.")
        except Exception as e:
            print(f"Error processing URL {url}: {e}, Status code: {response.status_code}")


if __name__ == "__main__":
    # Step 1: Fetch URLs (페이지 0부터 10까지 스크랩)
    urls, session_cookies = fetch_urls()

    # Step 2: Fetch and save content (모든 URL 처리)
    fetch_and_save_content(urls, session_cookies)
