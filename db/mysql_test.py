from os.path import dirname
import sys
sys.path.append(dirname(dirname(__file__)))
from db.mysql import mysql_client

if __name__ == '__main__':
    db = mysql_client.KnowledgeBaseManager()
    for p in db.fuzzy_get_poi_by_name('%天坛%'):
        print(str(p))