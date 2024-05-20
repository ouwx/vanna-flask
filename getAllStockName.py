import requests,json
import mysql.connector
import pandas as pd
from io import BytesIO

shbaseUrl="https://yunhq.sse.com.cn:32042/v1/sh1/list/exchange/ebs?callback=jsonpCallback94812691&select=code%2Cname%2Copen%2Chigh%2Clow%2Clast%2Cprev_close%2Cchg_rate%2Cvolume%2Camount%2Ccpxxextendname%2Ctradephase&_=1700481272043"
host = 'a.stockyun.top'
user = 'root'
password = '197911'
db_port = '3306'
database = 'stock'

def get_shETF_data():
    #url = shbaseUrl.format(page)
    response = requests.get(shbaseUrl)
    jsonp_data=response.text
    # 提取JSON数据
    json_data = json.loads(jsonp_data.split('(', 1)[1].rstrip(')'))

    # Accessing the 'list' key in the JSON data
    etf_list = json_data['list']
    date=json_data['date']

    save_shETF_data(etf_list,date)

def save_shETF_data(etf_list,date):
    # Connecting to MySQL
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        port=db_port,
        database=database
    )
    # Creating a cursor object
    cursor = conn.cursor()

    # SQL statement to insert data into the ETF table
    insert_query = """
    INSERT ignore shetf (
        ticker_symbol,etf_name, opening_price, closing_price, lowest_price, highest_price,
        last_traded_price, price_change_percentage, volume, turnover, etf_description, identifier,date
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # Inserting data into the ETF table
    for etf in etf_list:
        etf.append(date)
        print(etf)
        cursor.execute(insert_query, etf)
        # Committing the changes
        conn.commit()

    # Closing the cursor and connection
    cursor.close()
    conn.close()

def get_szETF_data():
    # URL to the Excel file
    url = "https://fund.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=fund_etf&tab1PAGENO=1&random=0.6931648431184001&TABKEY=tab1"

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Read the Excel file into a Pandas DataFrame
        df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
        for index, row in df.iterrows():
            zqdm_value = row["证券代码"]
            save_szETF_data(get_oneSZ_data(zqdm_value))
            print(f"证券代码 at index {index}: {zqdm_value}")

        # Display the DataFrame
        print(df.head())
    else:
        print("Failed to retrieve the Excel file. Status code:", response.status_code)

def save_szETF_data(df):
    # Connecting to MySQL
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        port=db_port,
        database=database
    )
    cursor = conn.cursor()
    print(df)

    # SQL statement to insert data into the ETF table
    insert_query = """
    INSERT ignore szetfprice  VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    df_list = [tuple(df.loc[i]) for i in range(len(df))]
    cursor.executemany(insert_query, df_list)
        # Committing the changes
    conn.commit()

    # Closing the cursor and connection
    cursor.close()
    conn.close()

def get_oneSZ_data(code):
    baseurl="https://fund.szse.cn/api/report/ShowReport/data?SHOWTYPE=JSON&CATALOGID=fund_jqhq&TABKEY=tab1&txtDm={}&random=0.3867221262802256"
    url = baseurl.format(code)
    response = requests.get(url)
    json_data=response.text
    data = json.loads(json_data)
    # Extract 'cols' and 'data' from the JSON
    cols = data[0]["metadata"]["cols"]
    extracted_data = data[0]["data"]

    # Convert extracted data to a Pandas DataFrame
    df = pd.DataFrame(extracted_data)

    # Rename columns using 'cols' values
    df.rename(columns=cols, inplace=True)

    # Display the DataFrame
    return (df)

if __name__ == "__main__":
    get_shETF_data()
    get_szETF_data()
    #print(data)

