@echo off
call activategap.bat
title Run flask
set FLASK_APP=app.py
set FLASK_ENV=development
set FLASK_DEBUG=0
flask run --port=5010 --host=0.0.0.0