from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# returns list of lists, where a sublist has the following format:
# [word, age_of_acquisition, concreteness, familiarity, imageability]
# '-' indicates no such psych value exists for the given word
def query_mrc_db(word_list):
    driver = webdriver.Chrome()
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