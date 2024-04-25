# from flask import Flask, jsonify
from agents.MemoryAgent import generate_response
from flask import Flask, jsonify, request, send_file,send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tools.neo4jVisual import generateCypher
app = Flask(__name__)
CORS(app)


# 定义一个路由，返回特定 ID 的图书
@app.route('/login', methods=['post'])
def login():
    print("接收到了")
    data = {
                "nickName": "哈哈哈",
                "userId": 123,
                "isAdmin": 0,
                "email": "2971133710@qq.com",
                "sex": 0,
                "personDescription": "哈哈哈",
                "joinTime": "2024",

                "status": 0

            }
    return jsonify({"status": "success", "code": 200, "info": "登录成功", "data": data})



from tools.SemanticMatching import semanticMatching
from tools.PictureDescripe import picturedescripe
from tools.JudgeLLM import Translatechain
from agents.MemoryAgent import history
from tools.SkinDetect import detectAndAnswer
import os
@app.route('/uploadPicAndAnswer', methods=['POST'])
def upload_file_answer():
    response=""
    if 'file' in request.files:
        file = request.files['file']
        content = request.form.get('content')
        
        if len(content)==0:
            content="请为我介绍一下这个皮肤病，以及这个皮肤病的预防措施，治疗方法"
        history.add_user_message(content)
        filepath_ = ""
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join("/root/data/flaskProject/SkinPictures/", filename)
            filepath_ = filepath
            file.save(filepath)
        flag=semanticMatching(filepath_)
        if flag==0:
            res=detectAndAnswer(filepath_,content)
            print(res)
            history.add_ai_message(res)
            response=res
            
        else:
            des=picturedescripe(filepath_)
            final=Translatechain.invoke({"input": des})
            history.add_ai_message(final)
            response=final
            print(final)


    else:
        content = request.form.get('content')
        response = generate_response(content)
        print(response)
    

    response_data = {'processed_content': f'{response}'}
    return jsonify(response_data)
@app.route('/getNeo4j', methods=['POST'])
def get_neo4j_data():
    if request.method == 'POST':
        ask_info = request.json.get('askInfo')  # 获取前端传递的参数

        # dat ="hhhh"
        dat=generateCypher(ask_info)
        response_data = {
            "status": "success",
            "code": 200,
            "info": "请求成功",
            "data": dat  # 文章ID
        }

        return jsonify(response_data)
if __name__ == '__main__':
    app.run(host="0.0.0.0")
