
import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'Top_SECRET_KEY'
    MAX_CONTENT_LENGTH= 24 * 1024 * 1024 * 1024  # 12GB 上傳限制
    UPLOAD_PATH= os.path.join(os.path.realpath(__package__), "uploads")