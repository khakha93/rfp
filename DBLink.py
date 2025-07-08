import psycopg2
from sshtunnel import SSHTunnelForwarder
import os
from dotenv import load_dotenv

load_dotenv()

def connect_db():
    # 환경 변수에서 연결 정보 로드
    ssh_host = os.getenv('DB_SSH_HOST')
    ssh_port = int(os.getenv('DB_SSH_PORT', 4633))
    ssh_username = os.getenv('DB_SSH_USERNAME')
    ssh_private_key_path = os.getenv('DB_SSH_PRIVATE_KEY')
    db_port = int(os.getenv('DB_PORT', 65432))
    db_password = os.getenv('DB_RFP_INFO_PASSWORD')

    # SSH 터널 연결
    ssh_tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_username,
        ssh_private_key=ssh_private_key_path,
        remote_bind_address=('127.0.0.1', db_port)
    )
    ssh_tunnel.start()
    
    # 데이터베이스 연결
    conn = psycopg2.connect(
        dbname='rfp_info',
        user='sdfseoul',
        password=db_password,
        host='127.0.0.1',
        port=ssh_tunnel.local_bind_port
    )
    
    return conn, ssh_tunnel

def close_db(conn, ssh_tunnel):
    cur = conn.cursor()
    cur.close()
    conn.close()
    ssh_tunnel.close()

# 테스트 코드
if __name__ == "__main__":
    conn, ssh_tunnel = connect_db()
    cur = conn.cursor()
    
    cur.execute("SELECT count(*) FROM public.bidding_information;")
    print(cur.fetchone())
    
    conn.commit()
    close_db(conn, ssh_tunnel)