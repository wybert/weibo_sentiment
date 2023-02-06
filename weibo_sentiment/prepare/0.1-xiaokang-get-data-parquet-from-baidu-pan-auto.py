import os
from pymongo import MongoClient
from tqdm import tqdm,trange
import pandas as pd
# import vaex
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

def download_data(could_file_path,download_path):
    cmd = """BaiduPCS-Go download %s "%s" -p 5 --save"""%(download_path,could_file_path)
    print(cmd)
    os.system(cmd)
    print("download data %s done!"%could_file_path)

def un_zip_data(file_name,unzip_outpath):
    """sudo apt update && sudo apt install --assume-yes p7zip-full #Ubuntu and Debian
    on mac:
    brew install p7zip
    use 7z insteda, eg:
    7z x hhh.7z
    """
    
    # cmd = """cd "%s"
    # 7za x %s"""%(unzip_outpath,file_name)
    cmd_win = """7z x %s -o%s"""%(file_name,unzip_outpath)
    print(cmd_win)
    os.system(cmd_win)
    print("unzip %s done"%file_name)

def upload_to_mongo(backup_path,file_name):

    cmd = """mongorestore --dir=%s%s"""%(backup_path,file_name.split(".")[0])
    print(cmd)
    os.system(cmd)
    print("upload %s done"%backup_path)

def get_collection_as_dataframe(collection,collection_name):
    data_list = []
    for item in tqdm(collection.find(),total = collection.estimated_document_count(),desc = "%s"%collection_name):
        temp = {}
        # print(item)
        temp["created_at"] = item["created_at"]
        temp["id"] = item["id"]
        temp["text"] = item["text"]
        temp["pic_ids"] = ",".join(item["pic_ids"])

        try:
            temp["bmiddle_pic"] = item["bmiddle_pic"]
            # break
        except Exception as e:
            # print(e)
            temp["bmiddle_pic"] = None

        try:
            [temp["lat"],temp["lon"]] = item["geo"]["coordinates"]
        except Exception as e:
            pass
            # print(e)
            # print(item["id"])
            # break
        temp["user_id"] = item["user"]["id"]
        temp["user_location"] = item["user"]["location"]
        temp["user_gender"] =  item["user"]["gender"]
        temp["user_followers_count"] = item["user"]["followers_count"]
        temp["user_friends_count"] = item["user"]["friends_count"]
        temp["user_statuses_count"] = item["user"]["statuses_count"]
        temp["user_lang"] = item["user"]["lang"]
        data_list += [temp]
        # break
        
    data_list = pd.DataFrame(data_list)

    return data_list
def save_coulumnar_files(root_output_path):
    
    mongo_client = MongoClient("mongodb://localhost:27017")

    root_path = root_output_path
    output_path1 = os.path.join(root_path,"weibo_output", "parquet")
    output_path2 = os.path.join(root_path,"weibo_output", "vaex_hdf5")
    os.makedirs(output_path1,exist_ok = True)
    os.makedirs(output_path2,exist_ok = True)



    dblist = [item for item in mongo_client.list_database_names() if "weibodata" in item]



    for db_name in dblist:
        print("processing database %s"%db_name)
        db = mongo_client[db_name]
        #数据库中的集合名字
        collist = [item for item in db.list_collection_names() if (item != "China_tmp") ]

        # break
        for collection_name in collist:

            mycol = db[collection_name]


            data=get_collection_as_dataframe(mycol,collection_name)
            data["id"] = data["id"].astype(np.int64)
            data["created_at"] =  pd.to_datetime(data['created_at'], errors='coerce')
            data["user_id"] =data["user_id"].astype(np.int64)
            data["lon"] =data["lon"].astype(float)
            data["lat"] =data["lat"].astype(float)

            data.to_parquet("%s/%s.parquet"%(output_path1,collection_name),engine='pyarrow' )
            
            # vaex_df = vaex.from_pandas(data, copy_index=False)  
            # vaex_df.export_hdf5("%s/%s_column.hdf5"%(output_path2,collection_name))    

            # df = pd.read_parquet(file_path,engine="pyarrow")
            #存储为csv格式
            # data.to_csv("D:/jianguoyun/code/testarrow/%s.csv"%collist[i],encoding="utf-8-sig",index=False)
            # parquet格式存储，因为数据量较大，需要先转化为字符串再存储
            # test = data.astype(str)
            # data = pd.DataFrame(test)
            # table = pa.Table.from_pandas(data)
            # pq.write_table(table, 'D:/jianguoyun/code/testarrow/%s.parquet'% collist[i])
            # break
        # break
    print("done!")

    print("save to local coulumnar_files done")

def delete_database(db_name):
    cmd = """mongo 127.0.0.1:27017/%s --eval "db.dropDatabase();" """%db_name
    # mongo_client = MongoClient("mongodb://localhost:27017")
    # mongo_client.drop_database(db_name)
    os.system(cmd)
    print("delete mongo database done")

def delete_unziped_files(download_path,file_name):
    cmd ="""rm -r %s%s"""%(download_path,file_name.split(".")[0])
    print(cmd)
    os.system(cmd)
    print("delete unziped files %s done"%file_name)


if __name__ == "__main__":
    start_date = datetime.date(2021,4,1)
    end_date = datetime.date(2022,10,1)
    could_base_path = "/data/China_weibo_data/mongo_data_backup/"
    download_path = "G:\\work\\weibo_process\\"

    # download_path = "/Users/kang/baiduyun_sync/work/weibo_data_analysis/"
    while start_date <= end_date:
        file_name = ''.join(start_date.isoformat().split("-"))[:-2]+".7z"
        if file_name not in ["202207.7z"]:
            start_date += relativedelta(months=+1)
            continue
        print(file_name,"*"*60)
        start_date += relativedelta(months=+1)
        # break
        could_file_path = "%s%s"%(could_base_path,file_name)
        print(could_file_path)
        # download_data(could_file_path,download_path)
        # un_zip_data(file_name,download_path)
        upload_to_mongo(download_path,file_name)
        save_coulumnar_files(root_output_path=download_path)
        mongo_client = MongoClient("mongodb://localhost:27017")
        dblist = [item for item in mongo_client.list_database_names() if "weibodata" in item]
        for db_name in dblist:
            delete_database(db_name=db_name)
        # delete_unziped_files(download_path,file_name)
       # break
        

