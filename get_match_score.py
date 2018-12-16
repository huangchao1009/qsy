# -*- encoding:utf-8 -*-
#from gensim.models.word2vec import Word2VecKeyedVectors
import re
import datetime
import matplotlib
import numpy as np
import pandas as pd
import jieba
'''
sex                    144 non-null object
birth                  143 non-null object
living                 122 non-null object
home                   122 non-null object
height                 122 non-null float64
accept_other_school    144 non-null int64
accept_work            144 non-null int64
school                 143 non-null object
graduated_at           141 non-null float64
is_student             144 non-null int64
degree                 144 non-null object
major                  140 non-null object
career                 14 non-null object
feature                144 non-null object
necessary              144 non-null object
des                    129 non-null object
bonus                  144 non-null object
uuid                   144 non-null object
'''
province_dict={"北京市":1,"天津市":2,"上海市":3,"重庆市":4,"河北省":5,"山西省":6,"辽宁省":7,"吉林省":8,"黑龙江省":9,
               "江苏省":10,"浙江省":11,"安徽省":12,"福建省":13,"江西省":14,"山东省":15,"河南省":16,"湖北省":17,
               "湖南省":18,"广东省":19,"海南省":20,"四川省":21,"贵州省":22,"云南省":23,"陕西省":24,"甘肃省":25,
               "青海省":26,"台湾省":27,"内蒙古自治区":28,"广西壮族自治区":29,"西藏自治区":30,"宁夏回族自治区":31,
               "新疆维吾尔自治区":32,"香港特别行政区":33,"澳门特别行政区":34}

province_lst={"北京":1,"天津":2,"上海":3,"重庆":4,"河北":5,"山西":6,"辽宁":7,"吉林":8,"黑龙江":9,
               "江苏":10,"浙江":11,"安徽":12,"福建":13,"江西":14,"山东":15,"河南":16,"湖北":17,
               "湖南":18,"广东":19,"海南":20,"四川":21,"贵州":22,"云南":23,"陕西":24,"甘肃":25,
               "青海":26,"台湾":27,"内蒙古":28,"广西":29,"西藏":30,"宁夏":31,
               "新疆":32,"香港":33,"澳门":34,"北方":35,"南方":36,"东北":34,"杭州":11,
              "江":10,"浙":11,"沪":3}
# province_lst=["北京","天津","上海","重庆","河北","山西","辽宁","吉林","黑龙江",
#                "江苏","浙江","安徽","福建","江西","山东","河南","湖北",
#                "湖南","广东","海南","四川","贵州","云南","陕西","甘肃",
#                "青海","台湾","内蒙古","广西","西藏","宁夏",
#                "新疆","香港","澳门","北方","南方","东北","杭州","江浙沪"]
degree_dict={"硕士":2,"博士":3,"学士":1,"本科":1,"大四":1,"保密":0}
major_lst1=["计算机","cs","软件","机","光","电","控","信"]
major_lst2=["生","化","理","工","环","能","材","动","高","车","工","医","农","军"]
major_lst3=["管理","会计","广告","金融","语","教育","法","文","政","史","税","心","哲","艺","音"]
year_now=datetime.datetime.now().year  #今年年份
def age_transform(x):
    try:
        return int(year_now)-int(x.strip().split("/")[0])
    except:
        return x
def cutword(x):
    try:
        return "|".join(jieba.cut(x))
    except:
        return x

def province_transform(x):
    try:
        return province_dict[x.strip().split(' ')[0]]
    except:
        return x
def major_transform(x):
    try:
        if len([i for i in major_lst1 if i in x])>0:
            return 1
        elif len([i for i in major_lst2 if i in x])>0:
            return 2
        elif len([i for i in major_lst3 if i in x])>0:
            return 3
        else:
            return 0
    except:
        return 0

def need_height(x):
    x=list(map(int,x))
    if len(x)==0:
        return -1
    for i in x:
        if i>=150 and i<=200: #身高范围
            return i
    return -1

def need_age(x):
    x=list(map(int,x))
    if len(x)==0:
        return ""
    for i in x:
        if i>=20 and i<=35:  #年龄范围
            return i
        if i >=80 and i<=100:  #出身年份
            return year_now-i-1900
        if i>=1980 and i<=2000: #出身年份
            return year_now-i
    return ""

def need_place(x):
    place=[str(province_lst[i]) for i in province_lst if i in x]
    if len(place)>0:
        return ','.join(place)
    else:
        return ""
def need_degree(x):
    if "硕" in x or "研究生" in x:
        return 1
    else:
        return 0

def place_match(x,y):
    if y is np.nan:
        return -1
    if str(x) not in str(y):
        return 0   #0表示地点不匹配
    else:
        return 1
def parse_hanzi(x):
    res=re.findall("[^A-Za-z0-9~!#$%&\'()*+,-./:;<=>?，@[\\]^_`{|} ,]", x)
    return "".join(res)

def load_dic(x):   #载入腾讯的词向量文件
    with open(r"C:\hc_python\qiushiyuan\Tencent_AILab_ChineseEmbedding.txt",'r') as f:
        vecdic={}
        f.readline()
        f.readline()
        for line in f:
            words=line.split()
            vector =map(float,words[1:])
            vecdic[words[0]]=np.array(vector)

def transform_to_vec(x):
    vecdic = {}  # 800万向量字典 np.array
    try:
        word_lst=x.strip().split('|')
        return sum([vecdic[i] for i in word_lst])/len(word_lst)
    except:
        return np.zeros([1,200])

def consin_distance(v1,v2):  #计算两个向量的余弦值
    try:
        vec0=np.dot(v1[0],v2[0])
        vec1=np.sqrt(np.sum(np.power(v1[0],2)))
        vec2=np.sqrt(np.sum(np.power(v2[0],2)))
        return vec0/(vec1*vec2)
    except:
        return 0

def first_step():
    data=pd.read_csv(r"C:\hc_python\qiushiyuan\data.csv",delimiter="\t")
    test=pd.read_csv(r"C:\hc_python\qiushiyuan\test.csv",delimiter="\t")
    #a_uuid=list(set(data["uuid"])-set(test["uuid1"])-set(test["uuid2"])) #没有出现在data里的uuid
    del data["career"]        #删除覆盖率低的特征
    del data["graduated_at"]  #删除冗余特征
    data['birth']=data['birth'].map(age_transform) #年龄转化为数值
    data.rename(columns={"birth":"age"},inplace=True)

    data["demand"] = data["necessary"].map(lambda x: re.findall('[0-9]{2,4}', x))  # 解析出身高年龄
    data["demand_height"] = data["demand"].apply(need_height)  # 身高要求
    data["demand_age"] = data["demand"].apply(need_age)  # 年龄要求
    del data["demand"]
    data["demand_place"] = data["necessary"].apply(need_place)  # 家乡要求
    data["demand_degree"] = data["necessary"].apply(need_degree)  # 学位要求

    data["necessary"]=data["necessary"].map(parse_hanzi).apply(cutword)              #解析出中文并分词
    data["des"]=data["des"].astype(str).map(parse_hanzi).apply(cutword)               #解析出中文并分词
    data["bonus"]=data["bonus"].map(parse_hanzi).apply(cutword)                       #解析出中文并分词
    data["feature"] = data["feature"].map(parse_hanzi).apply(cutword)                 # 解析出中文并分词

    data["necessary"]=data["necessary"].apply(transform_to_vec)   #转换为200维的词向量
    data["bonus"] = data["bonus"].apply(transform_to_vec)         #转换为200维的词向量
    data["your_feature"]= (data["necessary"]+ data["bonus"])/2    #把这两列向量相加作为对方特征
    del data["necessary"]
    del data["bonus"]

    data["des"] = data["des"].apply(transform_to_vec)             #转换为200维的词向量
    data["feature"] = data["feature"].apply(transform_to_vec)     #转换为200维的词向量
    data["my_feature"]=(data["des"]+data["feature"])/2            #把这两列向量相加作为自己特征
    del data["des"]
    del data["feature"]

    data["living"]=data["living"].fillna(-1)
    data["living"]=data["living"].apply(province_transform).map(lambda x:int(x) if x else -1)
    data["home"]=data["home"].fillna(-1)
    data["home"]=data["home"].apply(province_transform).map(lambda x:int(x) if x else -1)

    data["school"]=data["school"].apply(lambda x: 1 if x=="浙江大学" else 0 )
    data["degree"]=data["degree"].apply(lambda x: degree_dict[x] if x in degree_dict else -1)
    data["major"]=data["major"].apply(major_transform)
    data["sex"][data["sex"]=="female"]=0
    data["sex"][data["sex"]=="male"]=1

    test.rename(columns={"uuid1":"uuid"},inplace=True)
    test=pd.merge(test,data,on='uuid',how='left')
    data.rename(columns={"uuid":"uuid2"},inplace=True)
    test=pd.merge(test,data,on="uuid2",how="left")
    test.rename(columns={"uuid":"uuid_x"},inplace=True)
    test.rename(columns={"uuid2":"uuid_y"},inplace=True)
    test.to_csv("./test_hc.csv",index=False,encoding="utf-8")
    print("1st done!")
def second_step():
    test1=pd.read_csv("./test_hc.csv")

    test1["score"]=-1
    test1["score"][(test1["sex_x"].isnull())|(test1["sex_y"].isnull())]=0         #性别的数据非空
    test1["score"][test1["sex_x"]==test1["sex_y"]]=0   #性别相同不匹配，分数为0
    test1["score"][(test1["age_x"].isnull())|(test1["age_y"].isnull())]=0         #年龄的数据非空
    test1["score"][(test1["age_x"]<18)|(test1["age_x"]>35)|(test1["age_y"]<18)|(test1["age_y"]>35)]= 0  #非正常年龄范围
    test1["score"][abs(test1["age_x"]-test1["age_y"])>5]=0  #年龄差过大不匹配
    test1["score"][(test1["accept_other_school_x"]==0)&(test1["school_y"]==0)]=0  #不接受外校
    test1["score"][(test1["accept_other_school_y"]==0)&(test1["school_x"]==0)]=0  #不接受外校
    test1["score"][(test1["accept_work_x"]==0)&(test1["is_student_y"]==0)]=0      #不接受工作
    test1["score"][(test1["accept_work_y"]==0)&(test1["is_student_x"]==0)]=0      #不接受工作
    test1["score"][((test1["sex_x"]==0)&(test1["age_x"]>test1["age_y"]))]=0       #女身高小于男
    test1["score"][((test1["sex_y"]==0)&(test1["age_y"]>test1["age_x"]))]=0       #女身高小于男
    test1["score"][(test1["demand_degree_x"]==1)&(test1["degree_y"]<=1)]=0        #有学历要求
    test1["score"][(test1["demand_degree_y"]==1)&(test1["degree_x"]<=1)]=0        #有学历要求
    test1["score"][(test1["height_x"]<test1["demand_height_y"])|(test1["height_y"]<test1["demand_height_x"])]=0 #有身高要求
    test1["score"][(test1["age_x"]>test1["demand_age_y"])|(test1["age_y"]>test1["demand_age_x"])]=0 #有年龄要求
    test1["demand_place_x"]=test1.apply(lambda row: place_match(row["living_y"],row["demand_place_x"]),axis=1)
    test1["demand_place_y"]=test1.apply(lambda row: place_match(row["living_x"],row["demand_place_y"]),axis=1)
    test1["score"][(test1["demand_place_x"]==0)|(test1["demand_place_y"]==0)]=0  #有地点要求
    #求两者特征的匹配值（计算余弦值）
    test1["cosin_x"]=test1.apply(lambda row: consin_distance(row["your_feature_x"],row["my_feature_y"]),axis=1)
    test1["cosin_y"] = test1.apply(lambda row: consin_distance(row["your_feature_y"], row["my_feature_x"]), axis=1)
    test1["cosin_score"]=test1["cosin_x"]+test1["cosin_y"]
    del test1["cosin_x"]
    del test1["cosin_y"]
    #余弦值大于0.5的为1
    test1["score"][test1["score"]==-1][test1["cosin_score"]>=0.5]=1
    #800万的词向量文件小破本导不进去
    #剩下的就当做条件匹配吧 （ - -||)
    test1["score"][test1["score"]==-1]=1
    result=test1[["uuid_x","uuid_y","score"]].copy()
    result.rename(columns={"uuid_x":"uuid1","uuid_y":"uuid2"},inplace=True)
    result.to_csv("./result.csv",index=False)

# M_data=data[data.sex=="male"]   #69
# F_data=data[data.sex=="female"] #75
#first_step()
#second_step()
print ("done!")




