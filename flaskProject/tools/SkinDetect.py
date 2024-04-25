from alibabacloud_imageprocess20200320.client import Client
from alibabacloud_imageprocess20200320.models import DetectSkinDiseaseAdvanceRequest
from alibabacloud_tea_util.models import RuntimeOptions
from alibabacloud_tea_openapi.models import Config
from SearchDetailsOfSkinDisease import searchSkins
configApi = Config(
  access_key_id="xxxx",
  access_key_secret="xxxx",
  endpoint='imageprocess.cn-shanghai.aliyuncs.com',
  # 访问的域名对应的region
  region_id='cn-shanghai'
)
runtime_option = RuntimeOptions()
def detectAndAnswer(url:str,question:str):
    img = open(url, 'rb')
    requestApi = DetectSkinDiseaseAdvanceRequest()
    requestApi.url_object = img
    requestApi.org_id = " "
    requestApi.org_name = " "
    client = Client(configApi)
    response = client.detect_skin_disease_advance(requestApi, runtime_option)
    print(response.body.data)
    # print(type(response.body.data))
    dict_data = eval(str(response.body.data))

    img.close()
    input_="皮肤病诊断结果为"+str(dict_data["Results"])+"用户问题为："+question
    print(input_)
    caption=searchSkins(input_)
    print("caption为:")
    print(caption)
    return caption
# detectAndAnswer("/root/show/b.jpeg","请问他这是怎么了，这种病怎么治疗")