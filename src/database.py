import redis
import json
from sqlalchemy import create_engine, Integer, String, Column, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# MySQL 数据库定义
MYSQL_URL = "mysql+pymysql://root:123456@localhost:3306/rag_sql?charset=utf8mb4"
engine = create_engine(MYSQL_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class MySQLChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), index=True)
    role = Column(String(50))
    content = Column(Text)

class MySQLChatSummary(Base):
    __tablename__ = "chat_summaries"
    session_id = Column(String(255), primary_key=True)
    summary = Column(Text)

# 自动在MySQL中创建表
Base.metadata.create_all(bind=engine)

# 分层记忆管理
class MemoryManager:
    def __init__(self,session_id: str, summary_llm, retain_n_turns=3):
        self.db_session = SessionLocal()
        self.session_id = session_id
        self.summary_llm = summary_llm
        self.retain_n_turns = retain_n_turns
        #初始化redis连接
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0,decode_responses=True)
        self.redis_key = f"chat_hot:{session_id}"

    def get_context_messages(self) ->list:
        """
        在每次提问前调用：从 MySQL 获取摘要，从 Redis 获取近期记录，拼装成上下文。
        """
        messages = []
        # 1. 从 MySQL 获取摘要
        summary_record = self.db_session.query(MySQLChatSummary).filter_by(session_id=self.session_id).first()
        if summary_record and summary_record.summary:
            messages.append(SystemMessage(content=f"这是之前的对话总结：{summary_record.summary}"))
        
        # 2. 从 Redis 获取近期记录
        hot_records = self.redis_client.lrange(self.redis_key, 0, -1)
        for record in hot_records:
            msg_data = json.loads(record)
            if msg_data["role"] == "human":
                messages.append(HumanMessage(content=msg_data["content"]))
            else:
                messages.append(AIMessage(content=msg_data["content"]))
                
        return messages
        
    def save_context(self, human_text:str, ai_text:str):
        """
        在模型回答完之后调用：双写 MySQL 和 Redis，如果 Redis 超长，则触发滚动压缩。
        """
        # 1. 双写MySQL（归档全量记录）
        self.db_session.add(MySQLChatLog(session_id=self.session_id, role="human", content=human_text))
        self.db_session.add(MySQLChatLog(session_id=self.session_id, role="ai", content=ai_text))
        self.db_session.commit()

        # 2. 双写Redis（维护近期记录）
        self._push_to_redis("human", human_text)
        self._push_to_redis("ai", ai_text)

        # 3. 检查 Redis 长度，触发滚动压缩
        # N 轮 = N*2 条消息
        max_redis_len = self.retain_n_turns * 2
        current_len = self.redis_client.llen(self.redis_key)
        
        if current_len > max_redis_len:
            # Redis 超载了！把最老的 2 条消息弹出来 (LPOP)
            old_human = json.loads(self.redis_client.lpop(self.redis_key))["content"]
            old_ai = json.loads(self.redis_client.lpop(self.redis_key))["content"]
            
            # 触发异步的滚动压缩逻辑
            self._compress_old_messages(old_human, old_ai)

    def _push_to_redis(self, role:str, content:str):
        # 存入 Redis 列表的右侧 (尾部)
        msg_json = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        self.redis_client.rpush(self.redis_key, msg_json)
        # 给热数据设置个过期时间（比如24小时不聊自动清空Redis，但MySQL里永远在）
        self.redis_client.expire(self.redis_key, 60*60*24)
        
    def _compress_old_messages(self, popped_human: str, popped_ai: str):
        """
        读取 MySQL 旧摘要 -> 与刚才弹出的废弃对话合并 -> 用大模型写出新摘要 -> 覆盖 MySQL
        """
        summary_record = self.db_session.query(MySQLChatSummary).filter_by(session_id=self.session_id).first()
        old_summary = summary_record.summary if summary_record else "无"

        # 让大模型干活：合并新旧记忆
        prompt = f"""
        你是一个记忆整理助手。请结合【旧摘要】和【刚刚结束的一轮对话】，用一两句话生成一个更新后的【新摘要】。
        只保留重要事实、用户偏好和核心结论，剔除废话。
        【旧摘要】: {old_summary}
        【刚刚结束的对话】:
        用户: {popped_human}
        AI: {popped_ai}
        【请输出新摘要】:
        """
        new_summary = self.summary_llm.invoke(prompt).content
        # 更新 MySQL 中的摘要记录
        if summary_record:
            summary_record.summary = new_summary
        else:
            summary_record = MySQLChatSummary(session_id=self.session_id, summary=new_summary)
            self.db_session.add(summary_record)
        self.db_session.commit()
        print(f"已更新摘要: {new_summary}")
