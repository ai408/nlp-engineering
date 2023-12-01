from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config


config = Config() # 定义一个配置
config.max_connection_pool_size = 10 # 设置最大连接数
connection_pool = ConnectionPool() # 初始化连接池
# 如果给定的服务器是ok的，返回true，否则返回false
ok = connection_pool.init([('172.27.211.84', 9669)], config)

# 方式1：connection pool自己控制连接释放
# 从连接池中获取一个session
session = connection_pool.get_session('root', 'nebula')

session.execute('USE nba') # 选择space
result = session.execute('SHOW TAGS') # 展示tags
print(result) # 打印结果

session.release() # 释放session

# 方式2：Session Pool，session将自动释放
with connection_pool.session_context('root', 'nebula') as session:
    session.execute('USE nba')
    result = session.execute('SHOW TAGS')
    print(result)

# 关闭连接池
connection_pool.close()