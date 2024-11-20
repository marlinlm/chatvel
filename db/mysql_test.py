from os.path import dirname
import sys
sys.path.append(dirname(dirname(__file__)))
from db.mysql import mysql_client

if __name__ == '__main__':
    db = mysql_client.KnowledgeBaseManager()
    rs = db.get_documents_by_kb_id('desc_poi')
    print(str(rs))