@echo off
:loop
echo Starting FastAPI server with Uvicorn...
pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" --name fastapi-server
echo Server started.
echo Waiting for server to stop...
pm2 logs fastapi-server --lines 1000
echo Server stopped or crashed. Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
