import time

from dotenv import load_dotenv
from vanna import ValidationError
from vanna.openai import OpenAI_Chat
from flask_cors import CORS
from chromadb.utils import embedding_functions

load_dotenv()
from openai import OpenAI
from functools import wraps
from flask import Flask, jsonify, Response, request, redirect, url_for
import flask
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from cache import MemoryCache

app = Flask(__name__, static_url_path='')
CORS(app, supports_credentials=True)  # 启用凭据支持

# SETUP
cache = MemoryCache()
dbName = "sqlite"
# client = OpenAI(base_url="http://a.stockyun.top:3000/v1", api_key="sk-cDGWwKhlSK0kXEeP5c8891D5C30a40Ba89C2A9867202A9C6")
client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key="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")


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config)

    def generate_sql(self, question: str, **kwargs) -> str:
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        if ('error' in kwargs.keys() and 'sql' in kwargs.keys() and (kwargs.get('sql') is not None)):
            prompt.append({'role': 'assistant', 'content': kwargs.get("sql")})
            reflection_prompt = '''
                You were giving the following prompt:

                {full_prompt}

                This was your response:

                {llm_response}

                There was an error with the response, either in the output format or the query itself.

                Ensure that the following rules are satisfied when correcting your response:
                1. SQL is valid DuckDB SQL, given the provided metadata and the DuckDB querying rules
                2. The query SPECIFICALLY references the correct tables: employees.csv and purchases.csv, and those tables are properly aliased? (this is the most likely cause of failure)
                3. Response is in the correct format ({{sql: <sql_here>}} or {{"error": <explanation here>}}) with no additional text?
                4. All fields are appropriately named
                5. There are no unnecessary sub-queries
                6. ALL TABLES are aliased (extremely important)

                Rewrite the response and respond ONLY with the valid output format with no additional commentary

                '''.format(full_prompt=question, llm_response=kwargs.get("error"))
            prompt.append({'role': 'user', 'content': reflection_prompt})
            kwargs.pop("sql")
            kwargs.pop("error")
        # self.log(prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(llm_response)
        return self.extract_sql(llm_response)


vn = MyVanna(client=client, config={"model": "llama3-70b-8192", "db_name": dbName})
# vn = MyVanna(client=client, config={"model": "@cf/meta/llama-3-8b-instruct"})
# vn = VannaDefault(model='flytest', api_key="50a55fd0554a42a298ea820e625b41b3")
# 定义数据库连接参数字典
host = '38.181.56.128'
dbname = 'stock'
user = 'view'
password = 'view'
port = 3306


# vn.connect_to_mysql(host=host, dbname=dbname, user=user, password=password, port=port)
# vn.connect_to_sqlite('https://vanna.ai/Chinook.sqlite')
# NO NEED TO CHANGE ANYTHING BELOW THIS LINE
def requires_cache(fields):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            id = request.args.get('id')
            if (id == None and fields[0] == "sql"):
                id = args[0]
                cache.set(id=id, field='sql', value=args[1])
                args = {}

            if id is None:
                return jsonify({"type": "error", "error": "No id provided"})

            for field in fields:
                if cache.get(id=id, field=field) is None:
                    return jsonify({"type": "error", "error": f"No {field} found"})

            field_values = {field: cache.get(id=id, field=field) for field in fields}

            # Add the id to the field_values
            field_values['id'] = id

            return f(*args, **field_values, **kwargs)

        return decorated

    return decorator


@app.route('/')
def root():
    global dbName
    if request.args.get("dbName") is not None:
        dbName=request.args.get("dbName")
    global vn
    #device = "cuda" if torch.cuda.is_available() else "cpu"

    bge_embeddingFunction = embedding_functions.SentenceTransformerEmbeddingFunction("/AI-ModelScope/bge-large-zh-v1.5","cpu", normalize_embeddings=True)

    vn = MyVanna(client=client, config={"model": "llama3-8b-8192", "db_name": dbName,"embedding_function": bge_embeddingFunction})
    # 定义数据库连接参数字典
    host = '38.181.56.128'
    dbname = 'stock'
    user = 'view'
    password = 'view'
    port = 3306

    if dbName == 'stock':
        vn.connect_to_mysql(host=host, dbname=dbname, user=user, password=password, port=port)
    elif dbName == "sqlite":
        vn.connect_to_sqlite('https://vanna.ai/Chinook.sqlite')
    return app.send_static_file('index.html')


@app.route('/api/v0/generate_questions', methods=['GET'])
def generate_questions():
    return jsonify({
        "type": "question_list",
        "questions": vn.generate_questions(),
        "header": "Here are some questions you can ask:"
    })


@app.route('/api/v0/generate_sql', methods=['POST'])
def generate_sql():
    question = flask.request.json.get('question')
    # 记录开始时间
    start_time = time.time()

    if question is None:
        return jsonify({"type": "error", "error": "No question provided"})

    id = cache.generate_id(question=question)
    sql = None
    error = None
    df = None
    # 直接执行，设定重试次数，最大为5
    for retry in range(5):
        print("retry:", retry)
        try:
            sql = vn.generate_sql(question=question, error=error, sql=sql)
            df = run_sql(id, sql)
            break
        except ValidationError as e:
            print(e)
            error = str(e).replace('"', '\\"').replace("'", "\\'")
            sql = sql.replace('"', '\\"').replace("'", "\\'")
        except Exception as e:
            print(e)
            error = str(e).replace('"', '\\"').replace("'", "\\'")
            sql = sql.replace('"', '\\"').replace("'", "\\'")

    cache.set(id=id, field='question', value=question)
    cache.set(id=id, field='sql', value=sql)
    # 记录结束时间，并打印结束时间-开始时间
    end_time = time.time()
    print("generate_sql Time:", end_time - start_time)
    if df is not None:
        return jsonify(
            {
                "type": "sql",
                "id": id,
                "text": sql.strip(),
                "df": df,
            })
    else:
        return jsonify({"type": "error", "error": "无法找到匹配的SQL语言，请调整提问"})


@requires_cache(['sql'])
def run_sql(id: str, sql: str):
    # vn.connect_to_mysql(host=host, dbname=dbname, user=user, password=password, port=port)
    df = vn.run_sql(sql=sql)
    cache.set(id=id, field='df', value=df)
    return df.head(20).to_json(orient='records', date_format='iso', date_unit='s')


@app.route('/api/v0/run_sql', methods=['POST'])
@requires_cache(['sql'])
def run_sql_vanna(id: str, sql: str):
    # vn.connect_to_mysql(host=host, dbname=dbname, user=user, password=password, port=port)
    try:
        sql = flask.request.json.get('sql')
        df = vn.run_sql(sql=sql)
        cache.set(id=id, field='df', value=df)
        return jsonify(
            {
                "type": "df",
                "id": id,
                "df": df.head(20).to_json(orient='records', date_format='iso', date_unit='s'),
            })

    except ValidationError as e:
        return jsonify({"type": "error", "error": str(e)})
    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})


@app.route('/api/v0/download_csv', methods=['GET'])
@requires_cache(['df'])
def download_csv(id: str, df):
    csv = df.to_csv()

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                     f"attachment; filename={id}.csv"})


@app.route('/api/v0/generate_plotly_figure', methods=['GET'])
@requires_cache(['df', 'question', 'sql'])
def generate_plotly_figure(id: str, df, question, sql):
    try:
        code = vn.generate_plotly_code(question=question, sql=sql,
                                       df_metadata=f"Running df.dtypes gives:\n {df.dtypes}")
        fig = vn.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
        fig_json = fig.to_json()

        cache.set(id=id, field='fig_json', value=fig_json)

        return jsonify(
            {
                "type": "plotly_figure",
                "id": id,
                "fig": fig_json,
            })
    except Exception as e:
        # Print the stack trace
        import traceback
        traceback.print_exc()

        return jsonify({"type": "error", "error": str(e)})


@app.route('/api/v0/get_training_data', methods=['GET'])
def get_training_data():
    df = vn.get_training_data()

    return jsonify(
        {
            "type": "df",
            "id": "training_data",
            "df": df.head(25).to_json(orient='records'),
        })


@app.route('/api/v0/remove_training_data', methods=['POST'])
def remove_training_data():
    # Get id from the JSON body
    id = flask.request.json.get('id')

    if id is None:
        return jsonify({"type": "error", "error": "No id provided"})

    if vn.remove_training_data(id=id):
        return jsonify({"success": True})
    else:
        return jsonify({"type": "error", "error": "Couldn't remove training data"})


@app.route('/api/v0/train', methods=['POST'])
def add_training_data():
    sql = None
    ddl = None
    documentation = None
    training_data_type = flask.request.json.get('training_data_type')
    question = flask.request.json.get('question')
    content = flask.request.json.get('content')
    if training_data_type != 'sql':
        question = None
    if training_data_type == "sql":
        sql = content
    elif training_data_type == "ddl":
        ddl = content
    elif training_data_type == "documentation":
        documentation = content

    try:
        id = vn.train(question=question, sql=sql, ddl=ddl, documentation=documentation)

        return jsonify({"id": id})
    except Exception as e:
        print("TRAINING ERROR", e)
        return jsonify({"type": "error", "error": str(e)})


@app.route('/api/v0/generate_followup_questions', methods=['GET'])
@requires_cache(['df', 'question', 'sql'])
def generate_followup_questions(id: str, df, question, sql):
    followup_questions = vn.generate_followup_questions(question=question, sql=sql, df=df)

    cache.set(id=id, field='followup_questions', value=followup_questions)

    return jsonify(
        {
            "type": "question_list",
            "id": id,
            "questions": followup_questions,
            "header": "Here are some followup questions you can ask:"
        })


@app.route('/api/v0/load_question', methods=['GET'])
@requires_cache(['question', 'sql', 'df', 'fig_json', 'followup_questions'])
def load_question(id: str, question, sql, df, fig_json, followup_questions):
    try:
        return jsonify(
            {
                "type": "question_cache",
                "id": id,
                "question": question,
                "sql": sql,
                "df": df.head(10).to_json(orient='records'),
                "fig": fig_json,
                "followup_questions": followup_questions,
            })

    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})


@app.route('/api/v0/get_question_history', methods=['GET'])
def get_question_history():
    return jsonify({"type": "question_history", "questions": cache.get_all(field_list=['question'])})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)
