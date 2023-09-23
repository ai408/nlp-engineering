from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
import numpy as np
import pandas as pd


config = Config() # 定义一个配置
config.max_connection_pool_size = 10 # 设置最大连接数
connection_pool = ConnectionPool() # 初始化连接池
# 如果给定的服务器是ok的，返回true，否则返回false
ok = connection_pool.init([('172.27.211.84', 9669)], config)

vertex_player_df = pd.read_csv("C:/Users/Administrator/Downloads/dataset/dataset/vertex_player.csv", header=None, names=['player_id', 'age', 'name'])
vertex_team_df = pd.read_csv("C:/Users/Administrator/Downloads/dataset/dataset/vertex_team.csv", header=None, names=['team_id', 'name'])
edge_follow_df = pd.read_csv("C:/Users/Administrator/Downloads/dataset/dataset/edge_follow.csv", header=None, names=['player_id1', 'player_id2', 'degree'])
edge_serve_df = pd.read_csv("C:/Users/Administrator/Downloads/dataset/dataset/edge_serve.csv", header=None, names=['player_id', 'team_id', 'start_year', 'end_year'])

# Session Pool，session将自动释放
with connection_pool.session_context('root', 'nebula') as session:
    # 创建basketballplayer_python空间
    session.execute('CREATE SPACE IF NOT EXISTS `basketballplayer_python_test` (vid_type = FIXED_STRING(32))')
    # result = session.execute('SHOW SPACES')
    # print(result)

    # 使用basketballplayer_python空间
    session.execute('USE basketballplayer_python')

    session.execute('CREATE TAG IF NOT EXISTS player(name string, age int)') # 创建player标签
    session.execute('CREATE TAG IF NOT EXISTS team(name string)') # 创建team标签
    session.execute('CREATE EDGE IF NOT EXISTS follow(degree int)') # 创建follow边
    session.execute('CREATE EDGE IF NOT EXISTS serve(start_year int, end_year int)') # 创建serve边

    # 从CSV文件中读取数据，插入到player标签中
    for index, row in vertex_player_df.iterrows():
        session.execute('INSERT VERTEX IF NOT EXISTS player(name, age) VALUES "{}":("{}", {})'.format(row['player_id'], row['name'], np.int64(row['age'])))
    # 从CSV文件中读取数据，插入到team标签中
    for index, row in vertex_team_df.iterrows():
        session.execute('INSERT VERTEX IF NOT EXISTS team(name) VALUES "{}":("{}")'.format(row['team_id'], row['name']))
    # 从CSV文件中读取数据，插入到follow边中
    for index, row in edge_follow_df.iterrows():
        session.execute('INSERT EDGE IF NOT EXISTS follow(degree) VALUES "{}"->"{}":({})'.format(row['player_id1'], row['player_id2'], np.int64(row['degree'])))
    # 从CSV文件中读取数据，插入到serve边中
    for index, row in edge_serve_df.iterrows():
        session.execute('INSERT EDGE IF NOT EXISTS serve(start_year, end_year) VALUES "{}"->"{}":({}, {})'.format(row['player_id'], row['team_id'], np.int64(row['start_year']), np.int64(row['end_year'])))

# 关闭连接池
connection_pool.close()