@echo off
chcp 65001 > nul
title KEGS 실행기 (로컬 + 서버 요청)
echo ================================================
echo   🎮 KEGS - 게임 영상 자막 생성기 (Windows 자동 실행)
echo ================================================

REM ✅ 현재 bat 파일 기준 KEGS 루트로 이동
cd /d "%~dp0"

REM ✅ logs 폴더 생성 (루트에)
if not exist logs (
    mkdir logs
)

REM ✅ 로그 파일 절대 경로 저장
set LOG_PATH=%~dp0logs\backend_error.log

REM [💡 주의] 백엔드 에러 로그 초기화 (절대경로로)
echo. > "%LOG_PATH%"

REM ✅ 백엔드 실행 (venv 환경에서)
start "KEGS Backend" cmd /k "cd /d %~dp0backend && call ..\venv\Scripts\activate.bat && python -m uvicorn main:app --host 0.0.0.0 --port 8000 2> "%LOG_PATH%"

REM ✅ 프론트엔드 실행
start "KEGS Frontend" cmd /k "cd /d %~dp0frontend && python -m http.server 5500"

echo.
echo ✅ 백엔드 (localhost:8000) 와 프론트엔드 (localhost:5500)가 실행되었습니다.
echo 🔗 브라우저에서 http://localhost:5500 에 접속하세요.

pause
