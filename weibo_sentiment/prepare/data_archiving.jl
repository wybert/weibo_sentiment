using Mongoc
using DataFrames
using ProgressBars


# using DataFrames.MongoDB
function trans_doc(document)
    temp = Dict()
    temp["created_at"] = document["created_at"]
    temp["id"] = document["id"]
    temp["text"] = document["text"]
    temp["pic_ids"] = join(document["pic_ids"], ",")

    try
        temp["bmiddle_pic"] = document["bmiddle_pic"]
        # break
    catch e
        # print(e)
        temp["bmiddle_pic"] = nothing
    end

    try
        temp["lat"] =  document["geo"]["coordinates"][1]
        temp["lon"] =  document["geo"]["coordinates"][2]
    catch e
    
    end
        # pass
        # print(e)
        # print(document["id"])
        # break
    temp["user_id"] = document["user"]["id"]
    temp["user_location"] = document["user"]["location"]
    temp["user_gender"] =  document["user"]["gender"]
    temp["user_followers_count"] = document["user"]["followers_count"]
    temp["user_friends_count"] = document["user"]["friends_count"]
    temp["user_statuses_count"] = document["user"]["statuses_count"]

    return temp
end



client = Mongoc.Client("mongodb://localhost:27017")
# Mongoc.ping(client)
collection = client["weibodata2107"]["China_2107-08-14"]
# document = Mongoc.find_one(collection, Mongoc.BSON("""{  }"""))
# z = trans_doc(document)
a = []
for document in tqdm(collection)
    z = trans_doc(document)
    pop!(a,z)
    # break
end

df = DataFrame(;[Symbol(k) =>v for (k,v) in a]...)




# backup_path = "/Users/kang/Downloads/weibo_process/"
# file_name = "2107"
