create_environment:
	python -m venv env
enter_environment:
    ./env/Scripts/Activate.ps1
install_libraries:
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
make_requirementstxt:
    pip freeze > requirements.txt
update_policy:
	Set-ExecutionPolicy RemoteSigned
