import psycopg2

from os import system, path, getenv
import json

import random

import hashlib

from joblib import Parallel, delayed

from flask import Flask, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import segmentation
import trcolumns
import avec


# %%
"""
### DB connection
"""

# %%

# загружаем окружение из .env
load_dotenv()

params = {
    'database': getenv('DB_NAME'),
    'user':     getenv('DB_USER'),
    'password': getenv('DB_PASSWORD'),
    'host':     getenv('DB_HOST'),
    'port':     getenv('DB_PORT'),
}

conn = psycopg2.connect(**params)
cursor = conn.cursor()


app = Flask(__name__)


# %%
"""
### Загрузка файла для обработки
"""

# %%


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return json.dumps({'success': 'False'})

        file = request.files['file']

        if file.filename == '':
            return json.dumps({'success': 'False'})
        if file:
            filename = secure_filename(file.filename)
            file.save(filename)

        inputFile = filename
        directory = hashlib.md5(
            str(f'{inputFile}' + str(random.randint(1, 9999999))).encode('utf-8')).hexdigest()

        complete_result = []

        if path.isfile(inputFile):
            system(
                f'ffmpeg -y -loglevel quiet -i "{inputFile}" -ac 2 {directory}.wav')
            inputFile = f'{directory}.wav'
            if path.isfile(inputFile):

                filenames = segmentation(inputFile, directory)

                # Запускаем параллельную векторизацию сегментов
                results = Parallel(n_jobs=-1)(delayed(avec.audio2vec)(f)
                                              for f in filenames)

                for r in results:
                    if ('file' in r):
                        data = trcolumns.transform_columns(r)

                        # Выполняем поиск по Евклидовому расстоянию
                        cursor.execute(
                            'select id,file,vid, vector<->cube(ARRAY[{}]) from voices order by 4 limit 10'.format(data.at[0, 'vector']))

                        for row in cursor:
                            complete_result.append(
                                [row[1].replace("d:\\", ""), row[3]])

            else:
                return json.dumps({'status': 'Error'})

        else:
            json.dumps({'status': 'Error'})

        return json.dumps(complete_result)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9008)
