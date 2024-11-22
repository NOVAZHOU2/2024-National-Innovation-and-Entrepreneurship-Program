chcp 65001
@echo off

set start_time=%time%
echo =: 开始时间: %start_time%

echo =: 正在创建虚拟环境...
env\python -m venv workenv

echo =: 正在进入虚拟环境...
call workenv\Scripts\activate.bat
python.exe -m pip install --upgrade pip

echo =: 正在安装依赖...
REM 定义requirements.txt文件路径
set REQUIREMENTS=requirements.txt

REM 检查requirements.txt文件是否存在
if not exist %REQUIREMENTS% (
    echo =: requirements.txt 文件不存在
    exit /b 1
)

REM 读取requirements.txt文件中的所有库名称
for /f "tokens=1,* delims==" %%i in (%REQUIREMENTS%) do (
	REM 检查当前库是否已安装
    pip show %%i >nul 2>&1
    if errorlevel 1 (
        REM 如果未安装，则使用pip安装该库
        echo =: 安装 %%i... %time%
        pip install %%i
    ) else (
        echo =: %%i 已安装
    )
)

echo =: 所有依赖都已安装

set end_time=%time%
echo =: 结束时间: %end_time%

python -c "from datetime import datetime as dt; start_time = dt.strptime('%start_time%', '%%H:%%M:%%S.%%f'); end_time = dt.strptime('%end_time%', '%%H:%%M:%%S.%%f'); time_diff = end_time - start_time; print('环境配置完成，耗时:', time_diff)"
deactivate