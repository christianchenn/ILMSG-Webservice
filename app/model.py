from flask_sqlalchemy import SQLAlchemy
from app.database import db

class WordsSentences(db.Model):
    __tablename__ = 'Words_Sentences'
    wordId = db.Column(db.Integer, db.ForeignKey('words.id'), primary_key=True)
    sentenceId = db.Column(db.Integer, db.ForeignKey('sentences.id'), primary_key=True)
    sentence = db.relationship('Sentence', back_populates='words')
    word = db.relationship('Word', back_populates='sentences')

class Category(db.Model):
    __tablename__ = 'categories'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    name = db.Column(db.String(191))

class Recording(db.Model):
    __tablename__ = 'recording'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    started_at = db.Column(db.DateTime)
    stopped_at = db.Column(db.DateTime)
    sessionId = db.Column(db.Integer, db.ForeignKey('sessions.id'))
    sentenceId = db.Column(db.Integer, db.ForeignKey('sentences.id'))
    session = db.relationship('Session', back_populates='recording')
    sentence = db.relationship('Sentence', back_populates='recording')

class Sentence(db.Model):
    __tablename__ = 'sentences'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    text = db.Column(db.String(191))
    wordOrder = db.Column(db.String(191))
    recordings = db.relationship('Recording', back_populates='sentence')
    words = db.relationship('WordsSentences', back_populates='sentence')

class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    started_at = db.Column(db.DateTime)
    finished_at = db.Column(db.DateTime)
    speakerId = db.Column(db.Integer, db.ForeignKey('speakers.id'))
    recordings = db.relationship('Recording', back_populates='session')
    speaker = db.relationship('Speaker', back_populates='sessions')

class Speaker(db.Model):
    __tablename__ = 'speakers'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    name = db.Column(db.String(191))
    gender = db.Column(db.String(191))
    dob = db.Column(db.DateTime)
    sessions = db.relationship('Session', back_populates='speaker')

class Word(db.Model):
    __tablename__ = 'words'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    text = db.Column(db.String(191))
    syllables = db.Column(db.Integer)
    categoryId = db.Column(db.Integer, db.ForeignKey('categories.id'))
    category = db.relationship('Category')
    sentences = db.relationship('WordsSentences', back_populates='word')
