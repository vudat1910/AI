from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import csv

# Khởi tạo dịch vụ ChromeDriver
service = Service('C:/Users/dat.vuphat/Downloads/chromedriver-win64/chromedriver.exe')
driver = webdriver.Chrome(service=service)

# Truy cập vào trang web
driver.get('https://mobion.vn/home/introduction')

# Tìm bảng
table = driver.find_element(By.XPATH, '//table[@class="table table-bordered table-responsive"]')

# Lấy tất cả các hàng trong bảng
rows = table.find_elements(By.TAG_NAME, "tr")

# Mở file CSV để ghi
with open('data_plus.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Duyệt qua từng hàng và ghi vào file
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        data = [cell.text for cell in cells]
        
        # Chỉ ghi nếu hàng có dữ liệu
        if data:
            writer.writerow(data)

# Đóng driver
driver.quit()
