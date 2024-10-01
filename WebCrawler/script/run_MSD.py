import argparse
import json
import re
import time
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("window-size=1400,1500")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("start-maximized")
    options.add_argument("enable-automation")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    return driver

def extract_data(driver, url):
    driver.get(url)
    driver.implicitly_wait(10)

    div_element = driver.find_element(By.CLASS_NAME, "Symptoms_symptoms__main__Yy_M7")
    li_elements = div_element.find_elements(By.TAG_NAME, "li")

    data_list = []

    for index in tqdm(range(len(li_elements))):
        div_element = driver.find_element(By.CLASS_NAME, "Symptoms_symptoms__main__Yy_M7")
        li_elements = div_element.find_elements(By.TAG_NAME, "li")
        
        li = li_elements[index]
        link = li.find_element(By.TAG_NAME, "a") 
        href = link.get_attribute("href")
        name = link.text
        
        try:
            link.click()
        except:
            actions = ActionChains(driver).move_to_element(link)
            actions.perform()
            link.click()

        # Name extraction
        topic_head_element = driver.find_element(By.XPATH, '//*[contains(@class, "TopicHead_tablebox")]')
        h1_text = topic_head_element.find_element(By.TAG_NAME, 'h1').text

        try:
            h2_text = topic_head_element.find_element(By.TAG_NAME, 'h2').text
        except:
            h2_text = None

        if h2_text:
            h1_text = h1_text.strip()
            h2_text = h2_text.strip()
            h2_text = re.sub(r'[()]', '', h2_text)
            name = h1_text + ' / ' + h2_text
        else:
            name = h1_text.strip()

        category = '증상'

        # Explanation extraction
        explanation = []
        main_content = driver.find_element(By.CSS_SELECTOR, '[data-testid="topic-main-content"]')
        stop_words = ["원인", "평가", "치료", "요점"]
        for element in main_content.find_elements(By.XPATH, './*'):
            element_id = element.get_attribute('id')
            if element_id and any(word in element_id for word in stop_words):
                break
            if element.text.strip():
                explanation.append(element.text.strip())
        explanation = "\n".join(explanation)

        # Causes, evaluation, treatment, keypoints extraction
        causes, evaluation, treatment, keypoints = None, None, None, None

        sections = driver.find_elements(By.XPATH, "//section[contains(@id, '원인') or contains(@id, '평가') or contains(@id, '치료') or contains(@id, '요점')]")
        for section in sections:
            section_id = section.get_attribute('id')
            if "원인" in section_id:
                causes = section.text
            elif "평가" in section_id:
                evaluation = section.text
            elif "치료" in section_id:
                treatment = section.text
            elif "요점" in section_id:
                keypoints = section.text

        # Constructing output data
        data = {
            'name': name,
            'category': category,
            'explanation': explanation,
            'causes': causes,
            'evaluation': evaluation,
            'treatment': treatment,
            'keypoints': keypoints,
            'source': 'MSD',
            'url': href
        }
        data_list.append(data)

        driver.back()

    return data_list

def save_data(data_list, output_file):
    # Converting list of dictionaries to a DataFrame
    df = pd.DataFrame(data_list)

    # Saving to CSV file
    df.to_csv(output_file, index=False, encoding='utf-8')

def main(url, output_file):
    driver = setup_driver()
    data_list = extract_data(driver, url)
    save_data(data_list, output_file)
    driver.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSD Symptoms Data Crawler')
    parser.add_argument('--url', type=str, default='https://www.msdmanuals.com/ko-kr/home/symptoms', help='URL to start crawling from')
    parser.add_argument('--output', type=str, required=True, help='Output file path for the crawled data (CSV format)')

    args = parser.parse_args()
    main(args.url, args.output)
