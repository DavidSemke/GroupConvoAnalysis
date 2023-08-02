from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import numpy as np

# Parameter ratings is the output from func query_mrc_db
def avg_ratings(ratings):
    # cols and rows labels; each starts listing at col/row 0
    # 2 columns: propy_score_total, propy_word_total
    # 4 rows: aoa, cnc, fam, img
    psych_prop_matrix = np.zeros((4, 2), dtype=int)
    
    for rating in ratings:
        # index 0 is the word
        aoa = rating[1]
        cnc = rating[2]
        fam = rating[3]
        img = rating[4]

        props = [aoa, cnc, fam, img]
        for i in range(len(props)):
            
            if props[i] == '-': continue

            psych_prop_matrix[i][0] += int(props[i])
            psych_prop_matrix[i][1] += 1
    
    avgs = []
    for i in range(len(props)):
        avg = psych_prop_matrix[i][0]/psych_prop_matrix[i][1]
        avgs.append(avg)

    return avgs


# returns list of lists, where a sublist has the following format:
# [word, age_of_acquisition, concreteness, familiarity, imageability]
# '-' indicates no such psych value exists for the given word
# ratings do not include duplicate words 
def query_mrc_db(word_list):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    driver = webdriver.Chrome(options=options)
    driver.get('https://websites.psychology.uwa.edu.au/school/mrcdatabase/uwa_mrc.htm')

    # input to DB on main page
    section1_indexes = [3, 18, 20, 22]
    section1(driver, section1_indexes)

    word_pattern = " ".join(word_list)
    section3_tuples = [(4, word_pattern)]
    section3(driver, section3_tuples)

    # submit input on main page
    submit(driver)

    # get results from resulting page
    ratings = strip_results_from_HTML(driver)
    
    driver.quit()

    return ratings


def section1(driver, selection_indexes):
    field_checkboxes = driver.find_elements(By.NAME, 'o')
    # select age of acquisition, familiarity, concreteness, imageability
    for i in selection_indexes:
        field_checkboxes[i].click()


def section2(driver, minmax_tuples):
    # tuples have format (row_index, min, max)
    for (i, min, max) in minmax_tuples:
        min_name = "n" + str(i)
        max_name = "x" + str(i)
        min_text_input = driver.find_element(By.NAME, min_name)
        max_text_input = driver.find_element(By.NAME, max_name)
        min_text_input.send_keys(min)
        max_text_input.send_keys(max)


# ignores capitalization - pronunication variability filters
# starts at simple letter match
def section3(driver, pattern_tuples):
    # tuples have format (input_index, pattern)
    for (i, patt) in pattern_tuples:
        patt_name = "p" + str(i)
        patt_text_input = driver.find_element(By.NAME, patt_name)
        patt_text_input.send_keys(patt)


def submit(driver):
    driver.find_element(By.XPATH, "//input[@type='submit' and @value='GO']").click()


def strip_results_from_HTML(driver):
    wait = WebDriverWait(driver, timeout=10)
    pre_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'pre')))
    inner_html = pre_element.get_attribute('innerHTML')
    output = inner_html.split("<hr>", 1)[1].strip()
    rows = output.split('\n')
    for i in range(len(rows)):
        rows[i] = rows[i].split()
    
    return rows