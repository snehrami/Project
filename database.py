from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class AnalysisHistory(Base):
    __tablename__ = 'analysis_history'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    image_source = Column(String(50))
    description = Column(Text, nullable=True)
    full_report = Column(Text, nullable=False)

try:
    engine = create_engine('sqlite:///history.db')
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
except:
    pass

def save_analysis(image_source: str, full_report: str):
    session = SessionLocal()
    try:
        new_entry = AnalysisHistory(
            image_source=image_source,
            full_report=full_report
        )
        session.add(new_entry)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_history(limit=10):
    session = SessionLocal()
    try:
        return session.query(AnalysisHistory).order_by(AnalysisHistory.created_at.desc()).limit(limit).all()
    finally:
        session.close()
