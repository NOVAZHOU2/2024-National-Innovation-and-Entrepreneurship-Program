chcp 65001
@echo off

echo 注意：程序运行过程中切勿关闭本窗口，否则会导致程序中断运行！！！
echo 如果您是第一次启动本程序，可能您需要等待一会……

REM 定义requirements.txt文件路径
set WORKENV=workenv

REM 检查环境是否初始化
if not exist %WORKENV% (
    echo =: 环境未初始化，正在初始化环境，可能需要1-2分钟，请耐心等待...
    call init_workenv.bat
)
.\workenv\Scripts\python.exe newMain.py

pause