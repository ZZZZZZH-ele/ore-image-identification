from flask import Flask, request

app = Flask(__name__)

@app.route('/http://127.126.0.1:5000/', methods=['POST'])
def upload():
    if 'photo' not in request.files:
        return 'No file uploaded.', 400

    photo = request.files['photo']
    if photo.filename == '':
        return 'Invalid file.', 400

    # 处理上传的照片，例如保存到服务器或者进行进一步的处理
    
    # 这里只是简单地将照片保存到当前目录下的uploaded_photos文件夹中
    photo.save('uploaded_photos/' + photo.filename)

    return 'File uploaded successfully.'

if __name__ == '__main__':
    app.run()