import os
from flask import Flask, request, render_template, jsonify
import json

app = Flask(__name__)

# JSON dosyasının yolu
dosya_yolu = os.path.join('data','news.json')

# JSON dosyasını yükle
with open(dosya_yolu, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Anasayfa
@app.route('/')
def index():
    return render_template('index.html', data=json_data)

# JSON'u güncelle
@app.route('/update', methods=['POST'])
def update():
    data = request.json
    for item in json_data:
        if item['ID'] == data['ID']:
            item['Durum'] = data['Durum']
            break

    # JSON'u güncellenmiş haliyle dosyaya kaydet
    with open(dosya_yolu, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

    
    return jsonify({"message": "Durum başarıyla güncellendi!"}), 200

if __name__ == '__main__':
    app.run(debug=True)

#3391480